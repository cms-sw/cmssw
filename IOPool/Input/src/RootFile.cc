/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootFile.h"
#include "DuplicateChecker.h"
#include "ProvenanceAdaptor.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/GroupSelector.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "DataFormats/Common/interface/EDProduct.h"
//used for friendlyName translation
#include "FWCore/Utilities/interface/FriendlyName.h"

//used for backward compatibility
#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/EventAux.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"
#include "DataFormats/Provenance/interface/RunAux.h"
#include "DataFormats/Provenance/interface/RunLumiEntryInfo.h"

#include "TROOT.h"
#include "TClass.h"
#include "TFile.h"
#include "TTree.h"
#include "Rtypes.h"
#include <algorithm>

namespace edm {
//---------------------------------------------------------------------
  RootFile::RootFile(std::string const& fileName,
		     std::string const& catalogName,
		     ProcessConfiguration const& processConfiguration,
		     std::string const& logicalFileName,
		     boost::shared_ptr<TFile> filePtr,
		     RunNumber_t const& startAtRun,
		     LuminosityBlockNumber_t const& startAtLumi,
		     EventNumber_t const& startAtEvent,
		     unsigned int eventsToSkip,
		     std::vector<LuminosityBlockID> const& whichLumisToSkip,
		     int remainingEvents,
		     int remainingLumis,
		     unsigned int treeCacheSize,
                     int treeMaxVirtualSize,
		     InputSource::ProcessingMode processingMode,
		     int forcedRunOffset,
		     std::vector<EventID> const& whichEventsToProcess,
                     bool noEventSort,
		     GroupSelectorRules const& groupSelectorRules,
                     bool dropMergeable,
                     boost::shared_ptr<DuplicateChecker> duplicateChecker,
                     bool dropDescendants) :
      file_(fileName),
      logicalFile_(logicalFileName),
      catalog_(catalogName),
      processConfiguration_(processConfiguration),
      filePtr_(filePtr),
      fileFormatVersion_(),
      fid_(),
      fileIndexSharedPtr_(new FileIndex),
      fileIndex_(*fileIndexSharedPtr_),
      fileIndexBegin_(fileIndex_.begin()),
      fileIndexEnd_(fileIndexBegin_),
      fileIndexIter_(fileIndexBegin_),
      eventProcessHistoryIDs_(),
      eventProcessHistoryIter_(eventProcessHistoryIDs_.begin()),
      startAtRun_(startAtRun),
      startAtLumi_(startAtLumi),
      startAtEvent_(startAtEvent),
      eventsToSkip_(eventsToSkip),
      whichLumisToSkip_(whichLumisToSkip),
      whichEventsToProcess_(whichEventsToProcess),
      eventListIter_(whichEventsToProcess_.begin()),
      noEventSort_(noEventSort),
      fastClonable_(false),
      reportToken_(0),
      eventAux_(),
      lumiAux_(),
      runAux_(),
      eventTree_(filePtr_, InEvent),
      lumiTree_(filePtr_, InLumi),
      runTree_(filePtr_, InRun),
      treePointers_(),
      productRegistry_(),
      branchIDLists_(),
      processingMode_(processingMode),
      forcedRunOffset_(forcedRunOffset),
      newBranchToOldBranch_(),
      eventHistoryTree_(0),
      history_(new History),
      branchChildren_(new BranchChildren),
      duplicateChecker_(duplicateChecker),
      provenanceAdaptor_() {

    eventTree_.setCacheSize(treeCacheSize);

    eventTree_.setTreeMaxVirtualSize(treeMaxVirtualSize);
    lumiTree_.setTreeMaxVirtualSize(treeMaxVirtualSize);
    runTree_.setTreeMaxVirtualSize(treeMaxVirtualSize);

    treePointers_[InEvent] = &eventTree_;
    treePointers_[InLumi]  = &lumiTree_;
    treePointers_[InRun]   = &runTree_;

    setRefCoreStreamer(0, true); // backward compatibility

    // Read the metadata tree.
    TTree *metaDataTree = dynamic_cast<TTree *>(filePtr_->Get(poolNames::metaDataTreeName().c_str()));
    if (!metaDataTree)
      throw edm::Exception(errors::FileReadError) << "Could not find tree " << poolNames::metaDataTreeName()
							 << " in the input file.\n";

    // To keep things simple, we just read in every possible branch that exists.
    // We don't pay attention to which branches exist in which file format versions

    FileFormatVersion *fftPtr = &fileFormatVersion_;
    metaDataTree->SetBranchAddress(poolNames::fileFormatVersionBranchName().c_str(), &fftPtr);

    FileID *fidPtr = &fid_;
    if (metaDataTree->FindBranch(poolNames::fileIdentifierBranchName().c_str()) != 0) {
      metaDataTree->SetBranchAddress(poolNames::fileIdentifierBranchName().c_str(), &fidPtr);
    }

    FileIndex *findexPtr = &fileIndex_;
    if (metaDataTree->FindBranch(poolNames::fileIndexBranchName().c_str()) != 0) {
      metaDataTree->SetBranchAddress(poolNames::fileIndexBranchName().c_str(), &findexPtr);
    }

    // Need to read to a temporary registry so we can do a translation of the BranchKeys.
    // This preserves backward compatibility against friendly class name algorithm changes.
    ProductRegistry tempReg;
    ProductRegistry *ppReg = &tempReg;
    metaDataTree->SetBranchAddress(poolNames::productDescriptionBranchName().c_str(),(&ppReg));

    typedef std::map<ParameterSetID, ParameterSetBlob> PsetMap;
    PsetMap psetMap;
    PsetMap *psetMapPtr = &psetMap;
    metaDataTree->SetBranchAddress(poolNames::parameterSetMapBranchName().c_str(), &psetMapPtr);

    ProcessHistoryRegistry::collection_type pHistMap;
    ProcessHistoryRegistry::collection_type *pHistMapPtr = &pHistMap;
    metaDataTree->SetBranchAddress(poolNames::processHistoryMapBranchName().c_str(), &pHistMapPtr);

    ProcessConfigurationRegistry::collection_type pProcConfigMap;
    ProcessConfigurationRegistry::collection_type *pProcConfigMapPtr = &pProcConfigMap;
    if (metaDataTree->FindBranch(poolNames::processConfigurationBranchName().c_str()) != 0) {
      metaDataTree->SetBranchAddress(poolNames::processConfigurationBranchName().c_str(), &pProcConfigMapPtr);
    }

    std::auto_ptr<BranchIDListRegistry::collection_type> branchIDListsAPtr(new BranchIDListRegistry::collection_type);
    BranchIDListRegistry::collection_type *branchIDListsPtr = branchIDListsAPtr.get();
    if (metaDataTree->FindBranch(poolNames::branchIDListBranchName().c_str()) != 0) {
      metaDataTree->SetBranchAddress(poolNames::branchIDListBranchName().c_str(), &branchIDListsPtr);
    }

    BranchChildren* branchChildrenBuffer = branchChildren_.get();
    if (metaDataTree->FindBranch(poolNames::productDependenciesBranchName().c_str()) != 0) {
      metaDataTree->SetBranchAddress(poolNames::productDependenciesBranchName().c_str(), &branchChildrenBuffer);
    }

    // backward compatibility
    std::vector<EventProcessHistoryID> *eventHistoryIDsPtr = &eventProcessHistoryIDs_;
    if (metaDataTree->FindBranch(poolNames::eventHistoryBranchName().c_str()) != 0) {
      metaDataTree->SetBranchAddress(poolNames::eventHistoryBranchName().c_str(), &eventHistoryIDsPtr);
    }

    ModuleDescriptionRegistry::collection_type mdMap;
    ModuleDescriptionRegistry::collection_type *mdMapPtr = &mdMap;
    if (metaDataTree->FindBranch(poolNames::moduleDescriptionMapBranchName().c_str()) != 0) {
      metaDataTree->SetBranchAddress(poolNames::moduleDescriptionMapBranchName().c_str(), &mdMapPtr);
    }
    // Here we read the metadata tree
    input::getEntry(metaDataTree, 0);

    setRefCoreStreamer(true);  // backward compatibility

    // Merge into the parameter set registry.
    pset::Registry& psetRegistry = *pset::Registry::instance();
    for (PsetMap::const_iterator i = psetMap.begin(), iEnd = psetMap.end(); i != iEnd; ++i) {
      ParameterSet pset(i->second.pset_);
      pset.setID(i->first);
      pset.setFullyTracked();
      psetRegistry.insertMapped(pset);
    } 

    if (fileFormatVersion_.value_ < 11) {
      // Old format input file.  Create a provenance adaptor.
      provenanceAdaptor_.reset(new ProvenanceAdaptor(tempReg, pHistMap, pProcConfigMap));
      // Fill in the branchIDLists branch from the provenance adaptor
      branchIDLists_ = provenanceAdaptor_->branchIDLists();
    } else {
      // New format input file. The branchIDLists branch was read directly from the input file. 
      if (metaDataTree->FindBranch(poolNames::branchIDListBranchName().c_str()) == 0) {
	throw edm::Exception(errors::EventCorruption)
	  << "Failed to find branchIDLists branch in metaData tree.\n";
      }
      branchIDLists_.reset(branchIDListsAPtr.release());
    }

    // Merge into the remaining hashed registries.
    ProcessConfigurationRegistry::instance()->insertCollection(pProcConfigMap);
    ProcessHistoryRegistry::instance()->insertCollection(pHistMap);
    ModuleDescriptionRegistry::instance()->insertCollection(mdMap);

    validateFile();

    // Read the parentage tree.  Old format files are handled internally in readParentageTree().
    readParentageTree();

    initializeDuplicateChecker();
    if (noEventSort_) fileIndex_.sortBy_Run_Lumi_EventEntry();
    fileIndexIter_ = fileIndexBegin_ = fileIndex_.begin();
    fileIndexEnd_ = fileIndex_.end();
    eventProcessHistoryIter_ = eventProcessHistoryIDs_.begin();

    readEventHistoryTree();

    // Set product presence information in the product registry.
    ProductRegistry::ProductList const& pList = tempReg.productList();
    for (ProductRegistry::ProductList::const_iterator it = pList.begin(), itEnd = pList.end();
        it != itEnd; ++it) {
      BranchDescription const& prod = it->second;
      treePointers_[prod.branchType()]->setPresence(prod);
    }

    // freeze our temporary product registry
    tempReg.setFrozen();

    std::auto_ptr<ProductRegistry> newReg(new ProductRegistry);

    // Do the translation from the old registry to the new one
    {
      ProductRegistry::ProductList const& prodList = tempReg.productList();
      for (ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
           it != itEnd; ++it) {
        BranchDescription const& prod = it->second;
        std::string newFriendlyName = friendlyname::friendlyName(prod.className());
	if (newFriendlyName == prod.friendlyClassName()) {
          newReg->copyProduct(prod);
	} else {
          if (fileFormatVersion_.value_ >= 11) {
	    throw edm::Exception(errors::UnimplementedFeature)
	      << "Cannot change friendly class name algorithm without more development work\n"
	      << "to update BranchIDLists.  Contact the framework group.\n";
	  }
          BranchDescription newBD(prod);
          newBD.updateFriendlyClassName();
          newReg->copyProduct(newBD);
	  // Need to call init to get old branch name.
	  prod.init();
	  newBranchToOldBranch_.insert(std::make_pair(newBD.branchName(), prod.branchName()));
	}
      }
      // freeze the product registry
      newReg->setFrozen();
      productRegistry_.reset(newReg.release());
    }

    dropOnInput(groupSelectorRules, dropDescendants, dropMergeable);

    // Set up information from the product registry.
    ProductRegistry::ProductList const& prodList = productRegistry()->productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
        it != itEnd; ++it) {
      BranchDescription const& prod = it->second;
      treePointers_[prod.branchType()]->addBranch(it->first, prod,
						  newBranchToOldBranch(prod.branchName()));
    }

    // Sort the EventID list the user supplied so that we can assume it is time ordered
    sort_all(whichEventsToProcess_);
    // Determine if this file is fast clonable.
    fastClonable_ = setIfFastClonable(remainingEvents, remainingLumis);

    reportOpened();
  }

  RootFile::~RootFile() {
  }

  void
  RootFile::readEntryDescriptionTree() {
    // Called only for old format files.
    if (fileFormatVersion_.value_ < 8) return; 
    TTree* entryDescriptionTree = dynamic_cast<TTree*>(filePtr_->Get(poolNames::entryDescriptionTreeName().c_str()));
    if (!entryDescriptionTree) 
      throw edm::Exception(errors::FileReadError) << "Could not find tree " << poolNames::entryDescriptionTreeName()
							 << " in the input file.\n";


    EntryDescriptionID idBuffer;
    EntryDescriptionID* pidBuffer = &idBuffer;
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionIDBranchName().c_str(), &pidBuffer);

    EntryDescriptionRegistry& oldregistry = *EntryDescriptionRegistry::instance();

    EventEntryDescription entryDescriptionBuffer;
    EventEntryDescription *pEntryDescriptionBuffer = &entryDescriptionBuffer;
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionBranchName().c_str(), &pEntryDescriptionBuffer);

    // Fill in the parentage registry.
    ParentageRegistry& registry = *ParentageRegistry::instance();

    for (Long64_t i = 0, numEntries = entryDescriptionTree->GetEntries(); i < numEntries; ++i) {
      input::getEntry(entryDescriptionTree, i);
      if (idBuffer != entryDescriptionBuffer.id())
	throw edm::Exception(errors::EventCorruption) << "Corruption of EntryDescription tree detected.\n";
      oldregistry.insertMapped(entryDescriptionBuffer);
      Parentage parents;
      parents.parents() = entryDescriptionBuffer.parents();
      registry.insertMapped(parents);
    }
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionIDBranchName().c_str(), 0);
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionBranchName().c_str(), 0);
  }

  void
  RootFile::readParentageTree()
  { 
    if (fileFormatVersion_.value_ < 11) {
      // Old format file.
      readEntryDescriptionTree();
      return;
    }
    // New format file
    TTree* parentageTree = dynamic_cast<TTree*>(filePtr_->Get(poolNames::parentageTreeName().c_str()));
    if (!parentageTree) 
      throw edm::Exception(errors::FileReadError) << "Could not find tree " << poolNames::parentageTreeName()
							 << " in the input file.\n";

    ParentageID idBuffer;
    ParentageID* pidBuffer = &idBuffer;
    parentageTree->SetBranchAddress(poolNames::parentageIDBranchName().c_str(), &pidBuffer);

    Parentage parentageBuffer;
    Parentage *pParentageBuffer = &parentageBuffer;
    parentageTree->SetBranchAddress(poolNames::parentageBranchName().c_str(), &pParentageBuffer);

    ParentageRegistry& registry = *ParentageRegistry::instance();

    for (Long64_t i = 0, numEntries = parentageTree->GetEntries(); i < numEntries; ++i) {
      input::getEntry(parentageTree, i);
      if (idBuffer != parentageBuffer.id())
        throw edm::Exception(errors::EventCorruption) << "Corruption of Parentage tree detected.\n";
      registry.insertMapped(parentageBuffer);
    }
    parentageTree->SetBranchAddress(poolNames::parentageIDBranchName().c_str(), 0);
    parentageTree->SetBranchAddress(poolNames::parentageBranchName().c_str(), 0);
  }

  bool
  RootFile::setIfFastClonable(int remainingEvents, int remainingLumis) const {
    if (!fileFormatVersion_.fastCopyPossible()) return false; 
    if (!fileIndex_.allEventsInEntryOrder()) return false; 
    if (!whichEventsToProcess_.empty()) return false; 
    if (eventsToSkip_ != 0) return false; 
    if (remainingEvents >= 0 && eventTree_.entries() > remainingEvents) return false;
    if (remainingLumis >= 0 && lumiTree_.entries() > remainingLumis) return false;
    if (processingMode_ != InputSource::RunsLumisAndEvents) return false; 
    if (forcedRunOffset_ != 0) return false; 
    // Find entry for first event in file
    FileIndex::const_iterator it = fileIndexBegin_;
    while(it != fileIndexEnd_ && it->getEntryType() != FileIndex::kEvent) {
      ++it;
    }
    if (it == fileIndexEnd_) return false;
    if (startAtRun_ > it->run_) return false;
    if (startAtRun_ == it->run_) {
      if (startAtLumi_ > it->lumi_) return false;
      if (startAtEvent_ > it->event_) return false;
    }
    for (std::vector<LuminosityBlockID>::const_iterator it = whichLumisToSkip_.begin(),
	  itEnd = whichLumisToSkip_.end(); it != itEnd; ++it) {
        if (fileIndex_.findLumiPosition(it->run(), it->luminosityBlock(), true) != fileIndexEnd_) {     
	  // We must skip a luminosity block in this file.  We will simply assume that
	  // it may contain an event, in which case we cannot fast copy.
	  return false;
        }
    }
    return true;
  }


  int
  RootFile::setForcedRunOffset(RunNumber_t const& forcedRunNumber) {
    if (fileIndexBegin_ == fileIndexEnd_) return 0;
    int defaultOffset = (fileIndexBegin_->run_ != 0 ? 0 : 1);
    forcedRunOffset_ = (forcedRunNumber != 0U ? forcedRunNumber - fileIndexBegin_->run_ : defaultOffset);
    if (forcedRunOffset_ != 0) {
      fastClonable_ = false;
    }
    return forcedRunOffset_;
  }

  boost::shared_ptr<FileBlock>
  RootFile::createFileBlock() const {
    return boost::shared_ptr<FileBlock>(new FileBlock(fileFormatVersion_,
						     eventTree_.tree(),
						     eventTree_.metaTree(),
						     lumiTree_.tree(),
						     lumiTree_.metaTree(),
						     runTree_.tree(),
						     runTree_.metaTree(),
						     fastClonable(),
						     file_,
						     branchChildren_));
  }

  std::string const&
  RootFile::newBranchToOldBranch(std::string const& newBranch) const {
    std::map<std::string, std::string>::const_iterator it = newBranchToOldBranch_.find(newBranch);
    if (it != newBranchToOldBranch_.end()) {
      return it->second;
    }
    return newBranch;
  }

  FileIndex::EntryType
  RootFile::getEntryType() const {
    if (fileIndexIter_ == fileIndexEnd_) {
      return FileIndex::kEnd;
    }
    return fileIndexIter_->getEntryType();
  }

  // Temporary KLUDGE until we can properly merge runs and lumis across files
  // This KLUDGE skips duplicate run or lumi entries.
  FileIndex::EntryType
  RootFile::getEntryTypeSkippingDups() {
    if (fileIndexIter_ == fileIndexEnd_) {
      return FileIndex::kEnd;
    }
    if (fileIndexIter_->event_ == 0 && fileIndexIter_ != fileIndexBegin_) {
      if ((fileIndexIter_-1)->run_ == fileIndexIter_->run_ && (fileIndexIter_-1)->lumi_ == fileIndexIter_->lumi_) {
	++fileIndexIter_;
	return getEntryTypeSkippingDups();
      } 
    }
    return fileIndexIter_->getEntryType();
  }

  FileIndex::EntryType
  RootFile::getNextEntryTypeWanted() {
    bool specifiedEvents = !whichEventsToProcess_.empty();
    if (specifiedEvents && eventListIter_ == whichEventsToProcess_.end()) {
      // We are processing specified events, and we are done with them.
      fileIndexIter_ = fileIndexEnd_;
      return FileIndex::kEnd;
    }
    FileIndex::EntryType entryType = getEntryTypeSkippingDups();
    if (entryType == FileIndex::kEnd) {
      return FileIndex::kEnd;
    }
    RunNumber_t const& currentRun = fileIndexIter_->run_;
    RunNumber_t correctedCurrentRun = (currentRun ? currentRun : 1U);
    if (specifiedEvents) {
       // We are processing specified events.
      if (correctedCurrentRun > eventListIter_->run()) {
	// The next specified event is in a run not in the file or already passed.  Skip the event
	++eventListIter_;
	return getNextEntryTypeWanted();
      }
      // Skip any runs before the next specified event.
      if (correctedCurrentRun < eventListIter_->run()) {
	fileIndexIter_ = fileIndex_.findRunPosition(eventListIter_->run(), false);      
	return getNextEntryTypeWanted();
      }
    }
    if (entryType == FileIndex::kRun) {
      // Skip any runs before the first run specified, startAtRun_.
      if (correctedCurrentRun < startAtRun_) {
        fileIndexIter_ = fileIndex_.findRunPosition(startAtRun_, false);      
	return getNextEntryTypeWanted();
      }
      return FileIndex::kRun;
    } else if (processingMode_ == InputSource::Runs) {
      fileIndexIter_ = fileIndex_.findRunPosition(currentRun + 1, false);      
      return getNextEntryTypeWanted();
    }
    LuminosityBlockNumber_t const& currentLumi = fileIndexIter_->lumi_;
    if (specifiedEvents) {
      // We are processing specified events.
      assert (correctedCurrentRun == eventListIter_->run());
      // Get the luminosity block number of the next specified event.
      FileIndex::const_iterator iter = fileIndex_.findEventPosition(currentRun, 0U, eventListIter_->event(), true);      
      if (iter == fileIndexEnd_ || currentLumi > iter->lumi_) {
	// Event Not Found or already passed. Skip the next specified event;
	++eventListIter_;
	return getNextEntryTypeWanted();
      }
      // Skip any lumis before the next specified event.
      if (currentLumi < iter->lumi_) {
        fileIndexIter_ = fileIndex_.findPosition(eventListIter_->run(), iter->lumi_, 0U);
        return getNextEntryTypeWanted();
      }
    }
    if (entryType == FileIndex::kLumi) {
      // Skip any lumis before the first lumi specified, startAtLumi_.
      assert(correctedCurrentRun >= startAtRun_);
      if (correctedCurrentRun == startAtRun_ && currentLumi < startAtLumi_) {
        fileIndexIter_ = fileIndex_.findLumiOrRunPosition(currentRun, startAtLumi_);      
	return getNextEntryTypeWanted();
      }
      // Skip the lumi if it is in whichLumisToSkip_.
      if (binary_search_all(whichLumisToSkip_, LuminosityBlockID(correctedCurrentRun, currentLumi))) {
        fileIndexIter_ = fileIndex_.findLumiOrRunPosition(currentRun, currentLumi + 1);      
	return getNextEntryTypeWanted();
      }
      return FileIndex::kLumi;
    } else if (processingMode_ == InputSource::RunsAndLumis) {
      fileIndexIter_ = fileIndex_.findLumiOrRunPosition(currentRun, currentLumi + 1);      
      return getNextEntryTypeWanted();
    }
    assert (entryType == FileIndex::kEvent);
    // Skip any events before the first event specified, startAtEvent_.
    assert(correctedCurrentRun >= startAtRun_);
    assert(correctedCurrentRun > startAtRun_ || currentLumi >= startAtLumi_);
    if (correctedCurrentRun == startAtRun_ &&
	fileIndexIter_->event_ < startAtEvent_) {
      fileIndexIter_ = fileIndex_.findPosition(currentRun, currentLumi, startAtEvent_);      
      return getNextEntryTypeWanted();
    }
    if (specifiedEvents) {
      // We have specified events to process and we've already positioned the file 
      // to execute the run and lumi entry for the current event in the list.
      // Just position to the right event.
      assert (correctedCurrentRun == eventListIter_->run());
      fileIndexIter_ = fileIndex_.findEventPosition(currentRun, currentLumi,
						  eventListIter_->event(), 
						  false);
      if (fileIndexIter_->event_ != eventListIter_->event()) {
	// Event was not found.
	++eventListIter_;
	return getNextEntryTypeWanted();
      }
      // Event was found.
      // For the next time around move to the next specified event
      ++eventListIter_;

      if (duplicateChecker_.get() != 0 &&
          duplicateChecker_->isDuplicateAndCheckActive(EventID(fileIndexIter_->run_, fileIndexIter_->event_),
                                                       fileIndexIter_->lumi_,
                                                       file_)) {
        ++fileIndexIter_;
        return getNextEntryTypeWanted();
      }

      if (eventsToSkip_ != 0) {
	// We have specified a count of events to skip.  So decrement the count and skip this event.
        --eventsToSkip_;
	return getNextEntryTypeWanted();
      }

      return FileIndex::kEvent;
    }

    if (duplicateChecker_.get() != 0 &&
        duplicateChecker_->isDuplicateAndCheckActive(EventID(fileIndexIter_->run_, fileIndexIter_->event_),
                                                     fileIndexIter_->lumi_,
                                                     file_)) {
      ++fileIndexIter_;
      return getNextEntryTypeWanted();
    }

    if (eventsToSkip_ != 0) {
      // We have specified a count of events to skip, keep skipping events in this lumi block
      // until we reach the end of the lumi block or the full count of the number of events to skip.
      while (eventsToSkip_ != 0 && fileIndexIter_ != fileIndexEnd_ &&
	getEntryTypeSkippingDups() == FileIndex::kEvent) {
        ++fileIndexIter_;
        --eventsToSkip_;

        while (
          eventsToSkip_ != 0 &&
          fileIndexIter_ != fileIndexEnd_ &&
          fileIndexIter_->getEntryType() == FileIndex::kEvent &&
          duplicateChecker_.get() != 0 &&
          duplicateChecker_->isDuplicateAndCheckActive(EventID(fileIndexIter_->run_, fileIndexIter_->event_),
                                                       fileIndexIter_->lumi_,
                                                       file_)) {
          ++fileIndexIter_;
        }
      }
      return getNextEntryTypeWanted();
    }
    return FileIndex::kEvent;
  }

  void
  RootFile::fillFileIndex() {
    // This function is for backward compatibility only.
    // Newer files store the file index.
    LuminosityBlockNumber_t lastLumi = 0;
    RunNumber_t lastRun = 0;

    // Loop over event entries and fill the index from the event auxiliary branch
    while(eventTree_.next()) {
      fillEventAuxiliary();
      fileIndex_.addEntry(eventAux_.run(),
			  eventAux_.luminosityBlock(),
			  eventAux_.event(),
			  eventTree_.entryNumber());
      // If the lumi tree is invalid, use the event tree to add lumi index entries.
      if (!lumiTree_.isValid()) {
	if (lastLumi != eventAux_.luminosityBlock()) {
	  lastLumi = eventAux_.luminosityBlock();
          fileIndex_.addEntry(eventAux_.run(), eventAux_.luminosityBlock(), 0U, FileIndex::Element::invalidEntry);
	}
      }
      // If the run tree is invalid, use the event tree to add run index entries.
      if (!runTree_.isValid()) {
	if (lastRun != eventAux_.run()) {
	  lastRun = eventAux_.run();
          fileIndex_.addEntry(eventAux_.run(), 0U, 0U, FileIndex::Element::invalidEntry);
        }
      }
    }
    eventTree_.setEntryNumber(-1);

    // Loop over luminosity block entries and fill the index from the lumi auxiliary branch
    if (lumiTree_.isValid()) {
      while (lumiTree_.next()) {
        fillLumiAuxiliary();
        fileIndex_.addEntry(lumiAux_.run(), lumiAux_.luminosityBlock(), 0U, lumiTree_.entryNumber());
      }
      lumiTree_.setEntryNumber(-1);
    }

    // Loop over run entries and fill the index from the run auxiliary branch
    if (runTree_.isValid()) {
      while (runTree_.next()) {
        fillRunAuxiliary();
        fileIndex_.addEntry(runAux_.run(), 0U, 0U, runTree_.entryNumber());
      }
      runTree_.setEntryNumber(-1);
    }
    fileIndex_.sortBy_Run_Lumi_Event();
  }

  void
  RootFile::validateFile() {
    if (!fileFormatVersion_.isValid()) {
      fileFormatVersion_.value_ = 0;
    }
    if (!fid_.isValid()) {
      fid_ = FileID(createGlobalIdentifier());
    }
    if(!eventTree_.isValid()) {
      throw edm::Exception(errors::EventCorruption) <<
	 "'Events' tree is corrupted or not present\n" << "in the input file.\n";
    }
    if (fileIndex_.empty()) {
      fillFileIndex();
    }
  }

  void
  RootFile::reportOpened() {
    // Report file opened.
    std::string const label = "source";
    std::string moduleName = "PoolSource";
    Service<JobReport> reportSvc;
    reportToken_ = reportSvc->inputFileOpened(file_,
               logicalFile_,
               catalog_,
               moduleName,
               label,
	       fid_.fid(),
               eventTree_.branchNames()); 
  }

  void
  RootFile::close(bool reallyClose) {
    if (reallyClose) {
      filePtr_->Close();
    }
    Service<JobReport> reportSvc;
    reportSvc->inputFileClosed(reportToken_);
  }

  void
  RootFile::fillEventAuxiliary() {
    if (fileFormatVersion_.value_ >= 3) {
      EventAuxiliary *pEvAux = &eventAux_;
      eventTree_.fillAux<EventAuxiliary>(pEvAux);
    } else {
      // for backward compatibility.
      EventAux eventAux;
      EventAux *pEvAux = &eventAux;
      eventTree_.fillAux<EventAux>(pEvAux);
      conversion(eventAux, eventAux_);
    }
    if (eventAux_.luminosityBlock_ == 0 && fileFormatVersion_.value_ <= 3) {
      eventAux_.luminosityBlock_ = LuminosityBlockNumber_t(1);
    } else if (fileFormatVersion_.value_ <= 1) {
      eventAux_.luminosityBlock_ = LuminosityBlockNumber_t(1);
    }
  }

  void
  RootFile::fillHistory() {
    // We could consider doing delayed reading, but because we have to
    // store this History object in a different tree than the event
    // data tree, this is too hard to do in this first version.

    if (fileFormatVersion_.value_ >= 7) {
      History* pHistory = history_.get();
      TBranch* eventHistoryBranch = eventHistoryTree_->GetBranch(poolNames::eventHistoryBranchName().c_str());
      if (!eventHistoryBranch)
	throw edm::Exception(errors::EventCorruption)
	  << "Failed to find history branch in event history tree.\n";
      eventHistoryBranch->SetAddress(&pHistory);
      input::getEntry(eventHistoryTree_, eventTree_.entryNumber());
    } else {
      // for backward compatibility.  If we could figure out how many
      // processes this event has been through, we should fill in
      // history_ with that many default-constructed IDs.
      if (!eventProcessHistoryIDs_.empty()) {
        if (eventProcessHistoryIter_->eventID_ != eventAux_.id()) {
          EventProcessHistoryID target(eventAux_.id(), ProcessHistoryID());
          eventProcessHistoryIter_ = lower_bound_all(eventProcessHistoryIDs_, target);	
          assert(eventProcessHistoryIter_->eventID_ == eventAux_.id());
        }
	history_->setProcessHistoryID(eventProcessHistoryIter_->processHistoryID_);
        ++eventProcessHistoryIter_;
      }
    }
    if (fileFormatVersion_.value_ < 11) {
      // old format.  branchListIndexes_ must be filled in from the ProvenanceAdaptor.
      provenanceAdaptor_->branchListIndexes(history_->branchListIndexes());
    }
  }

  void
  RootFile::fillLumiAuxiliary() {
    if (fileFormatVersion_.value_ >= 3) {
      LuminosityBlockAuxiliary *pLumiAux = &lumiAux_;
      lumiTree_.fillAux<LuminosityBlockAuxiliary>(pLumiAux);
    } else {
      LuminosityBlockAux lumiAux;
      LuminosityBlockAux *pLumiAux = &lumiAux;
      lumiTree_.fillAux<LuminosityBlockAux>(pLumiAux);
      conversion(lumiAux, lumiAux_);
    }
    if (lumiAux_.luminosityBlock() == 0 && fileFormatVersion_.value_ <= 3) {
      lumiAux_.id_ = LuminosityBlockID(lumiAux_.run(), LuminosityBlockNumber_t(1));
    }
  }

  void
  RootFile::fillRunAuxiliary() {
    if (fileFormatVersion_.value_ >= 3) {
      RunAuxiliary *pRunAux = &runAux_;
      runTree_.fillAux<RunAuxiliary>(pRunAux);
    } else {
      RunAux runAux;
      RunAux *pRunAux = &runAux;
      runTree_.fillAux<RunAux>(pRunAux);
      conversion(runAux, runAux_);
    }
  }

  int
  RootFile::skipEvents(int offset) {
    while (offset > 0 && fileIndexIter_ != fileIndexEnd_) {
      if (fileIndexIter_->getEntryType() == FileIndex::kEvent) {
        --offset;
      }
      ++fileIndexIter_;
    }
    while (offset < 0 && fileIndexIter_ != fileIndexBegin_) {
      --fileIndexIter_;
      if (fileIndexIter_->getEntryType() == FileIndex::kEvent) {
        ++offset;
      }
    }
    while (fileIndexIter_ != fileIndexEnd_ && fileIndexIter_->getEntryType() != FileIndex::kEvent) {
      ++fileIndexIter_;
    }
    return offset;
  }

  // readEvent() is responsible for creating, and setting up, the
  // EventPrincipal.
  //
  //   1. create an EventPrincipal with a unique EventID
  //   2. For each entry in the provenance, put in one Group,
  //      holding the Provenance for the corresponding EDProduct.
  //   3. set up the caches in the EventPrincipal to know about this
  //      Group.
  //
  // We do *not* create the EDProduct instance (the equivalent of reading
  // the branch containing this EDProduct. That will be done by the Delayed Reader,
  //  when it is asked to do so.
  //
  std::auto_ptr<EventPrincipal>
  RootFile::readEvent(boost::shared_ptr<ProductRegistry const> pReg) {
    assert(fileIndexIter_ != fileIndexEnd_);
    assert(fileIndexIter_->getEntryType() == FileIndex::kEvent);
    RunNumber_t currentRun = (fileIndexIter_->run_ ? fileIndexIter_->run_ : 1U);
    assert(currentRun >= startAtRun_);
    assert(currentRun > startAtRun_ || fileIndexIter_->lumi_ >= startAtLumi_);
    assert(currentRun > startAtRun_ || fileIndexIter_->lumi_ > startAtLumi_ ||
	 fileIndexIter_->event_ >= startAtEvent_);
    // Set the entry in the tree, and read the event at that entry.
    eventTree_.setEntryNumber(fileIndexIter_->entry_); 
    std::auto_ptr<EventPrincipal> ep = readCurrentEvent(pReg);

    assert(ep.get() != 0);
    assert(eventAux_.run() == fileIndexIter_->run_ + forcedRunOffset_);
    assert(eventAux_.luminosityBlock() == fileIndexIter_->lumi_);

    // report event read from file
    Service<JobReport> reportSvc;
    reportSvc->eventReadFromFile(reportToken_, eventID().run(), eventID().event());
    ++fileIndexIter_;
    return ep;
  }

  // Reads event at the current entry in the tree.
  // Note: This function neither uses nor sets fileIndexIter_.
  std::auto_ptr<EventPrincipal>
  RootFile::readCurrentEvent(boost::shared_ptr<ProductRegistry const> pReg) {
    if (!eventTree_.current()) {
      return std::auto_ptr<EventPrincipal>(0);
    }
    fillEventAuxiliary();
    fillHistory();
    overrideRunNumber(eventAux_.id_, eventAux_.isRealData());

    boost::shared_ptr<BranchMapper> mapper = (fileFormatVersion().value_ >= 11 ?
        makeBranchMapper<ProductProvenance>(eventTree_, InEvent) :
        makeBranchMapper<EventEntryInfo>(eventTree_, InEvent));

    // We're not done ... so prepare the EventPrincipal
    std::auto_ptr<EventPrincipal> thisEvent(new EventPrincipal(
		eventAux_,
		pReg,
		processConfiguration_,
		history_,
		mapper,
		eventTree_.makeDelayedReader(fileFormatVersion_.value_ < 11)));

    // Create a group in the event for each product
    eventTree_.fillGroups(*thisEvent);
    if (fileFormatVersion().value_ < 11 && fileFormatVersion().value_ >= 8) {
      thisEvent->readProvenanceImmediate();
    }
    return thisEvent;
  }

  void
  RootFile::setAtEventEntry(FileIndex::EntryNumber_t entry) {
    eventTree_.setEntryNumber(entry);
  }

  boost::shared_ptr<RunPrincipal>
  RootFile::readRun(boost::shared_ptr<ProductRegistry const> pReg) {
    assert(fileIndexIter_ != fileIndexEnd_);
    assert(fileIndexIter_->getEntryType() == FileIndex::kRun);
    RunNumber_t currentRun = (fileIndexIter_->run_ ? fileIndexIter_->run_ : 1U);
    assert(currentRun >= startAtRun_);
    // Begin code for backward compatibility before the exixtence of run trees.
    if (!runTree_.isValid()) {
      // prior to the support of run trees.
      // RunAuxiliary did not contain a valid timestamp.  Take it from the next event.
      if (eventTree_.next()) {
        fillEventAuxiliary();
        // back up, so event will not be skipped.
        eventTree_.previous();
      }
      RunID run = RunID(fileIndexIter_->run_);
      overrideRunNumber(run);
      ++fileIndexIter_;
      RunAuxiliary runAux(run.run(), eventAux_.time(), Timestamp::invalidTimestamp());
      return boost::shared_ptr<RunPrincipal>(
          new RunPrincipal(runAux,
	  pReg,
	  processConfiguration_));
    }
    // End code for backward compatibility before the exixtence of run trees.
    runTree_.setEntryNumber(fileIndexIter_->entry_); 
    fillRunAuxiliary();
    assert(runAux_.run() == fileIndexIter_->run_);
    overrideRunNumber(runAux_.id_);
    Service<JobReport> reportSvc;
    reportSvc->reportInputRunNumber(runAux_.run());
    if (runAux_.beginTime() == Timestamp::invalidTimestamp()) {
      // RunAuxiliary did not contain a valid timestamp.  Take it from the next event.
      if (eventTree_.next()) {
        fillEventAuxiliary();
        // back up, so event will not be skipped.
        eventTree_.previous();
      }
      runAux_.beginTime_ = eventAux_.time(); 
      runAux_.endTime_ = Timestamp::invalidTimestamp();
    }
    boost::shared_ptr<RunPrincipal> thisRun(
	new RunPrincipal(runAux_,
			 pReg,
			 processConfiguration_,
		         fileFormatVersion().value_ <= 10 && fileFormatVersion().value_ >= 8 ?
		         makeBranchMapper<RunLumiEntryInfo>(runTree_, InRun) :
		         makeBranchMapper<ProductProvenance>(runTree_, InRun),
			 runTree_.makeDelayedReader()));
    // Create a group in the run for each product
    runTree_.fillGroups(*thisRun);
    // Read in all the products now.
    thisRun->readImmediate();
    ++fileIndexIter_;
    return thisRun;
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  RootFile::readLumi(boost::shared_ptr<ProductRegistry const> pReg, boost::shared_ptr<RunPrincipal> rp) {
    assert(fileIndexIter_ != fileIndexEnd_);
    assert(fileIndexIter_->getEntryType() == FileIndex::kLumi);
    RunNumber_t currentRun = (fileIndexIter_->run_ ? fileIndexIter_->run_ : 1U);
    assert(currentRun >= startAtRun_);
    assert(currentRun > startAtRun_ || fileIndexIter_->lumi_ >= startAtLumi_);
    // Begin code for backward compatibility before the exixtence of lumi trees.
    if (!lumiTree_.isValid()) {
      if (eventTree_.next()) {
        fillEventAuxiliary();
        // back up, so event will not be skipped.
        eventTree_.previous();
      }

      LuminosityBlockID lumi = LuminosityBlockID(fileIndexIter_->run_, fileIndexIter_->lumi_);
      overrideRunNumber(lumi);
      ++fileIndexIter_;
      LuminosityBlockAuxiliary lumiAux(rp->run(), lumi.luminosityBlock(), eventAux_.time_, Timestamp::invalidTimestamp());
      return boost::shared_ptr<LuminosityBlockPrincipal>(
	new LuminosityBlockPrincipal(lumiAux,
				     pReg,
				     processConfiguration_));
    }
    // End code for backward compatibility before the exixtence of lumi trees.
    lumiTree_.setEntryNumber(fileIndexIter_->entry_); 
    fillLumiAuxiliary();
    assert(lumiAux_.run() == fileIndexIter_->run_);
    assert(lumiAux_.luminosityBlock() == fileIndexIter_->lumi_);
    overrideRunNumber(lumiAux_.id_);
    assert(lumiAux_.run() == rp->run());
    Service<JobReport> reportSvc;
    reportSvc->reportInputLumiSection(lumiAux_.run(), lumiAux_.luminosityBlock());

    if (lumiAux_.beginTime() == Timestamp::invalidTimestamp()) {
      // LuminosityBlockAuxiliary did not contain a timestamp. Take it from the next event.
      if (eventTree_.next()) {
        fillEventAuxiliary();
        // back up, so event will not be skipped.
        eventTree_.previous();
      }
      lumiAux_.beginTime_ = eventAux_.time();
      lumiAux_.endTime_ = Timestamp::invalidTimestamp();
    }
    boost::shared_ptr<LuminosityBlockPrincipal> thisLumi(
	new LuminosityBlockPrincipal(lumiAux_,
				     pReg, processConfiguration_,
				     fileFormatVersion().value_ <= 10 && fileFormatVersion().value_ >= 8 ?
				     makeBranchMapper<RunLumiEntryInfo>(lumiTree_, InLumi) :
				     makeBranchMapper<ProductProvenance>(lumiTree_, InLumi),
				     lumiTree_.makeDelayedReader()));
    // Create a group in the lumi for each product
    lumiTree_.fillGroups(*thisLumi);
    // Read in all the products now.
    thisLumi->readImmediate();
    ++fileIndexIter_;
    return thisLumi;
  }

  bool
  RootFile::setEntryAtEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, bool exact) {
    fileIndexIter_ = fileIndex_.findEventPosition(run, lumi, event, exact);
    if (fileIndexIter_ == fileIndexEnd_) return false;
    eventTree_.setEntryNumber(fileIndexIter_->entry_);
    return true;
  }

  bool
  RootFile::setEntryAtLumi(LuminosityBlockID const& lumi) {
    fileIndexIter_ = fileIndex_.findLumiPosition(lumi.run(), lumi.luminosityBlock(), true);
    if (fileIndexIter_ == fileIndexEnd_) return false;
    lumiTree_.setEntryNumber(fileIndexIter_->entry_);
    return true;
  }

  bool
  RootFile::setEntryAtRun(RunID const& run) {
    fileIndexIter_ = fileIndex_.findRunPosition(run.run(), true);
    if (fileIndexIter_ == fileIndexEnd_) return false;
    runTree_.setEntryNumber(fileIndexIter_->entry_);
    return true;
  }

  void
  RootFile::overrideRunNumber(RunID & id) {
    if (forcedRunOffset_ != 0) {
      id = RunID(id.run() + forcedRunOffset_);
    } 
    if (id < RunID::firstValidRun()) id = RunID::firstValidRun();
  }

  void
  RootFile::overrideRunNumber(LuminosityBlockID & id) {
    if (forcedRunOffset_ != 0) {
      id = LuminosityBlockID(id.run() + forcedRunOffset_, id.luminosityBlock());
    } 
    if (RunID(id.run()) < RunID::firstValidRun()) id = LuminosityBlockID(RunID::firstValidRun().run(), id.luminosityBlock());
  }

  void
  RootFile::overrideRunNumber(EventID & id, bool isRealData) {
    if (forcedRunOffset_ != 0) {
      if (isRealData) {
        throw edm::Exception(errors::Configuration,"RootFile::RootFile()")
          << "The 'setRunNumber' parameter of PoolSource cannot be used with real data.\n";
      }
      id = EventID(id.run() + forcedRunOffset_, id.event());
    } 
    if (RunID(id.run()) < RunID::firstValidRun()) id = EventID(RunID::firstValidRun().run(), id.event());
  }

  
  void
  RootFile::readEventHistoryTree() {
    // Read in the event history tree, if we have one...
    if (fileFormatVersion_.value_ < 7) return; 
    eventHistoryTree_ = dynamic_cast<TTree*>(filePtr_->Get(poolNames::eventHistoryTreeName().c_str()));

    if (!eventHistoryTree_)
      throw edm::Exception(errors::EventCorruption)
	<< "Failed to find the event history tree.\n";
  }

  void
  RootFile::initializeDuplicateChecker() {
    if (duplicateChecker_.get() != 0) {
      if (eventTree_.next()) {
        fillEventAuxiliary();
        duplicateChecker_->init(eventAux_.isRealData(),
                                fileIndex_);
      }
      eventTree_.setEntryNumber(-1);
    }
  }

  void
  RootFile::dropOnInput (GroupSelectorRules const& rules, bool dropDescendants, bool dropMergeable) {
    // This is the selector for drop on input.
    GroupSelector groupSelector;
    groupSelector.initialize(rules, productRegistry()->allBranchDescriptions());

    ProductRegistry::ProductList& prodList = const_cast<ProductRegistry::ProductList&>(productRegistry()->productList());
    // Do drop on input. On the first pass, just fill in a set of branches to be dropped.
    std::set<BranchID> branchesToDrop;
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
        it != itEnd; ++it) {
      BranchDescription const& prod = it->second;
      if(!groupSelector.selected(prod)) {
        if (dropDescendants) {
          branchChildren_->appendToDescendants(prod.branchID(), branchesToDrop);
        } else {
          branchesToDrop.insert(prod.branchID());
        }
      }
    }

    // On this pass, actually drop the branches.
    std::set<BranchID>::const_iterator branchesToDropEnd = branchesToDrop.end();
    for (ProductRegistry::ProductList::iterator it = prodList.begin(), itEnd = prodList.end(); it != itEnd;) {
      BranchDescription const& prod = it->second;
      bool drop = branchesToDrop.find(prod.branchID()) != branchesToDropEnd;
      if(drop) {
	if (groupSelector.selected(prod)) {
          LogWarning("RootFile")
            << "Branch '" << prod.branchName() << "' is being dropped from the input\n"
            << "of file '" << file_ << "' because it is dependent on a branch\n" 
            << "that was explicitly dropped.\n";
	}
        treePointers_[prod.branchType()]->dropBranch(newBranchToOldBranch(prod.branchName()));
        ProductRegistry::ProductList::iterator icopy = it;
        ++it;
        prodList.erase(icopy);
      } else {
        ++it;
      }
    }

    // Drop on input mergeable run and lumi products, this needs to be invoked for
    // secondary file input
    if (dropMergeable) {
      for (ProductRegistry::ProductList::iterator it = prodList.begin(), itEnd = prodList.end(); it != itEnd;) {
        BranchDescription const& prod = it->second;
        if (prod.branchType() != InEvent) {
          TClass *cp = gROOT->GetClass(prod.wrappedName().c_str());
          boost::shared_ptr<EDProduct> dummy(static_cast<EDProduct *>(cp->New()));
          if (dummy->isMergeable()) {
            treePointers_[prod.branchType()]->dropBranch(newBranchToOldBranch(prod.branchName()));
            ProductRegistry::ProductList::iterator icopy = it;
            ++it;
            prodList.erase(icopy);
          } else {
            ++it;
          }
        }
        else ++it;
      }
    }
  }

  // backward compatibility
  boost::shared_ptr<BranchMapper>
  RootFile:: makeBranchMapperInOldRelease(RootTree & rootTree, BranchType const& type) const {
    if (fileFormatVersion_.value_ >= 7) {
      rootTree.fillStatus();
    } else { 
       LogWarning("RootFile")
         << "Backward compatibility not fully supported for reading files"
	 << " written in CMSSW_1_8_4 or prior releases in releaseCMSSW_3_0_0.\n";
    }
    if (type == InEvent) {
      boost::shared_ptr<BranchMapperWithReader<EventEntryInfo> > mapper(new BranchMapperWithReader<EventEntryInfo>(0, 0));
      mapper->setDelayedRead(false);
      for(ProductRegistry::ProductList::const_iterator it = productRegistry_->productList().begin(),
          itEnd = productRegistry_->productList().end(); it != itEnd; ++it) {
        if (type == it->second.branchType() && !it->second.transient()) {
	  if (fileFormatVersion_.value_ >= 7) {
	    input::BranchMap::const_iterator ix = rootTree.branches().find(it->first);
	    input::BranchInfo const& ib = ix->second;
	    TBranch *br = ib.provenanceBranch_;
            std::auto_ptr<EntryDescriptionID> pb(new EntryDescriptionID);
            EntryDescriptionID* ppb = pb.get();
            br->SetAddress(&ppb);
            input::getEntry(br, rootTree.entryNumber());
	    std::vector<ProductStatus>::size_type index = it->second.oldProductID().productIndex() - 1;
	    EventEntryInfo entry(it->second.branchID(), rootTree.productStatuses()[index], it->second.oldProductID(), *pb);
	    mapper->insert(entry.makeProductProvenance());
          } else {
	    TBranch *br = rootTree.branches().find(it->first)->second.provenanceBranch_;
	    std::auto_ptr<BranchEntryDescription> pb(new BranchEntryDescription);
	    BranchEntryDescription* ppb = pb.get();
	    br->SetAddress(&ppb);
	    input::getEntry(br, rootTree.entryNumber());
	    std::auto_ptr<EntryDescription> entryDesc = pb->convertToEntryDescription();
	    ProductStatus status = (ppb->creatorStatus() == BranchEntryDescription::Success ? productstatus::present() : productstatus::neverCreated());
	    EventEntryInfo entry(it->second.branchID(), status, it->second.oldProductID());
	    mapper->insert(entry.makeProductProvenance());
	  }
	  mapper->insertIntoMap(it->second.oldProductID(), it->second.branchID());
        }
      }
      return mapper;
    } else {
      boost::shared_ptr<BranchMapperWithReader<ProductProvenance> > mapper(new BranchMapperWithReader<ProductProvenance>(0, 0));
      mapper->setDelayedRead(false);
      for(ProductRegistry::ProductList::const_iterator it = productRegistry_->productList().begin(),
          itEnd = productRegistry_->productList().end(); it != itEnd; ++it) {
	if (type == it->second.branchType() && !it->second.transient()) {
	  if (fileFormatVersion_.value_ >= 7) {
	    input::BranchMap::const_iterator ix = rootTree.branches().find(it->first);
	    input::BranchInfo const& ib = ix->second;
	    TBranch *br = ib.provenanceBranch_;
            input::getEntry(br, rootTree.entryNumber());
	    std::vector<ProductStatus>::size_type index = it->second.oldProductID().productIndex() - 1;
	    ProductProvenance entry(it->second.branchID(), rootTree.productStatuses()[index]);
	    mapper->insert(entry);
	  } else {
	    TBranch *br = rootTree.branches().find(it->first)->second.provenanceBranch_;
	    std::auto_ptr<BranchEntryDescription> pb(new BranchEntryDescription);
	    BranchEntryDescription* ppb = pb.get();
	    br->SetAddress(&ppb);
	    input::getEntry(br, rootTree.entryNumber());
	    std::auto_ptr<EntryDescription> entryDesc = pb->convertToEntryDescription();
	    ProductStatus status = (ppb->creatorStatus() == BranchEntryDescription::Success ? productstatus::present() : productstatus::neverCreated());
	    ProductProvenance entry(it->second.branchID(), status);
	    mapper->insert(entry);
	  }
	}
      }
      return mapper;
    }
    return boost::shared_ptr<BranchMapper>();
  }
  // end backward compatibility
}
