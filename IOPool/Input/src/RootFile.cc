/*----------------------------------------------------------------------
$Id: RootFile.cc,v 1.109 2008/01/21 03:11:45 wmtan Exp $
----------------------------------------------------------------------*/

#include "RootFile.h"


#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
//used for friendlyName translation
#include "FWCore/Utilities/interface/FriendlyName.h"

//used for backward compatibility
#include "DataFormats/Provenance/interface/EventAux.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"
#include "DataFormats/Provenance/interface/RunAux.h"

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
		     int remainingEvents,
		     int forcedRunOffset) :
      file_(fileName),
      logicalFile_(logicalFileName),
      catalog_(catalogName),
      processConfiguration_(processConfiguration),
      filePtr_(filePtr),
      fileFormatVersion_(),
      fid_(),
      fileIndex_(),
      fileIndexBegin_(fileIndex_.begin()),
      fileIndexEnd_(fileIndexBegin_),
      fileIndexIter_(fileIndexBegin_),
      eventProcessHistoryIDs_(),
      eventProcessHistoryIter_(eventProcessHistoryIDs_.begin()),
      startAtRun_(startAtRun),
      startAtLumi_(startAtLumi),
      startAtEvent_(startAtEvent),
      eventsToSkip_(eventsToSkip),
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
      forcedRunOffset_(forcedRunOffset),
      newBranchToOldBranch_(),
      sortedNewBranchNames_(),
      oldBranchNames_() {
    treePointers_[InEvent] = &eventTree_;
    treePointers_[InLumi]  = &lumiTree_;
    treePointers_[InRun]   = &runTree_;

    // Set up buffers for registries.
    // Need to read to a temporary registry so we can do a translation of the BranchKeys.
    // This preserves backward compatibility against friendly class name algorithm changes.
    ProductRegistry tempReg;
    ProductRegistry *ppReg = &tempReg;
    typedef std::map<ParameterSetID, ParameterSetBlob> PsetMap;
    PsetMap psetMap;
    ProcessHistoryMap pHistMap;
    ModuleDescriptionMap mdMap;
    PsetMap *psetMapPtr = &psetMap;
    ProcessHistoryMap *pHistMapPtr = &pHistMap;
    ModuleDescriptionMap *mdMapPtr = &mdMap;
    FileFormatVersion *fftPtr = &fileFormatVersion_;
    FileID *fidPtr = &fid_;
    FileIndex *findexPtr = &fileIndex_;
    std::vector<EventProcessHistoryID> *eventHistoryIDsPtr = &eventProcessHistoryIDs_;

    // Read the metadata tree.
    TTree *metaDataTree = dynamic_cast<TTree *>(filePtr_->Get(poolNames::metaDataTreeName().c_str()));
    assert(metaDataTree != 0);

    metaDataTree->SetBranchAddress(poolNames::productDescriptionBranchName().c_str(),(&ppReg));
    metaDataTree->SetBranchAddress(poolNames::parameterSetMapBranchName().c_str(), &psetMapPtr);
    metaDataTree->SetBranchAddress(poolNames::processHistoryMapBranchName().c_str(), &pHistMapPtr);
    metaDataTree->SetBranchAddress(poolNames::moduleDescriptionMapBranchName().c_str(), &mdMapPtr);
    metaDataTree->SetBranchAddress(poolNames::fileFormatVersionBranchName().c_str(), &fftPtr);
    if (metaDataTree->FindBranch(poolNames::fileIdentifierBranchName().c_str()) != 0) {
      metaDataTree->SetBranchAddress(poolNames::fileIdentifierBranchName().c_str(), &fidPtr);
    }
    if (metaDataTree->FindBranch(poolNames::fileIndexBranchName().c_str()) != 0) {
      metaDataTree->SetBranchAddress(poolNames::fileIndexBranchName().c_str(), &findexPtr);
    }
    if (metaDataTree->FindBranch(poolNames::eventHistoryBranchName().c_str()) != 0) {
      metaDataTree->SetBranchAddress(poolNames::eventHistoryBranchName().c_str(), &eventHistoryIDsPtr);
    }

    metaDataTree->GetEntry(0);

    validateFile();
    fileIndexIter_ = fileIndexBegin_ = fileIndex_.begin();
    fileIndexEnd_ = fileIndex_.end();
    eventProcessHistoryIter_ = eventProcessHistoryIDs_.begin();

    // freeze our temporary product registry
    tempReg.setFrozen();

    ProductRegistry *newReg = new ProductRegistry;
    // Do the translation from the old registry to the new one
    {
      ProductRegistry::ProductList const& prodList = tempReg.productList();
      for (ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
           it != itEnd; ++it) {
        BranchDescription const& prod = it->second;
        std::string newFriendlyName = friendlyname::friendlyName(prod.className());
	if (newFriendlyName == prod.friendlyClassName_) {
	  prod.init();
          newReg->addProduct(prod);
	} else {
          BranchDescription newBD(prod);
          newBD.friendlyClassName_ = newFriendlyName;
	  newBD.init();
          newReg->addProduct(newBD);
	  // Need to call init to get old branch name.
	  prod.init();
	  newBranchToOldBranch_.insert(std::make_pair(newBD.branchName(), prod.branchName()));
	  if (newBD.branchType() == InEvent) {
	    sortedNewBranchNames_.push_back(newBD.branchName());
	    oldBranchNames_.push_back(prod.branchName());
	  }
	}
      }
      sort_all(sortedNewBranchNames_);
      // freeze the product registry
      newReg->setFrozen();
      productRegistry_ = boost::shared_ptr<ProductRegistry const>(newReg);
    }

    // Merge into the registries. For now, we do NOT merge the product registry.
    pset::Registry& psetRegistry = *pset::Registry::instance();
    for (PsetMap::const_iterator i = psetMap.begin(), iEnd = psetMap.end(); i != iEnd; ++i) {
      psetRegistry.insertMapped(ParameterSet(i->second.pset_));
    } 
    ProcessHistoryRegistry & processNameListRegistry = *ProcessHistoryRegistry::instance();
    for (ProcessHistoryMap::const_iterator j = pHistMap.begin(), jEnd = pHistMap.end(); j != jEnd; ++j) {
      processNameListRegistry.insertMapped(j->second);
    } 
    ModuleDescriptionRegistry & moduleDescriptionRegistry = *ModuleDescriptionRegistry::instance();
    for (ModuleDescriptionMap::const_iterator k = mdMap.begin(), kEnd = mdMap.end(); k != kEnd; ++k) {
      moduleDescriptionRegistry.insertMapped(k->second);
    } 

    // Set up information from the product registry.
    ProductRegistry::ProductList const& prodList = productRegistry()->productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
        it != itEnd; ++it) {
      BranchDescription const& prod = it->second;
      treePointers_[prod.branchType()]->addBranch(it->first, prod,
						 newBranchToOldBranch(prod.branchName()));
    }

    // Determine if this file is fast clonable.
    fastClonable_ = setIfFastClonable(remainingEvents);
    if (fileIndexIter_ != fileIndexEnd_) {
      RunNumber_t currentRun = (fileIndexIter_->run_ ? fileIndexIter_->run_ : 1U);
      if (currentRun < startAtRun_) {
        fileIndexIter_ = fileIndex_.findPosition(startAtRun_, 0U, 0U);      
      }
      assert(fileIndexIter_ == fileIndexEnd_ || fileIndexIter_->getEntryType() == FileIndex::kRun);
    }

    reportOpened();
  }

  RootFile::~RootFile() {
  }

  bool
  RootFile::setIfFastClonable(int remainingEvents) const {
    if (fileFormatVersion_.value_ < 3) return false; 
    if (!fileIndex_.eventsSorted()) return false; 
    if (eventsToSkip_ != 0) return false; 
    if (remainingEvents >= 0 && eventTree_.entries() > remainingEvents) return false;
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
						     sortedNewBranchNames_,
						     oldBranchNames_));
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

  void
  RootFile::fillFileIndex() {
    // This function is for backward compatibility only.
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
          fileIndex_.addEntry(eventAux_.run(), eventAux_.luminosityBlock(), 0U, -1LL);
	}
      }
      // If the run tree is invalid, use the event tree to add run index entries.
      if (!runTree_.isValid()) {
	if (lastRun != eventAux_.run()) {
	  lastRun = eventAux_.run();
          fileIndex_.addEntry(eventAux_.run(), 0U, 0U, -1LL);
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
    fileIndex_.sort();
  }

  void
  RootFile::validateFile() {
    if (!fileFormatVersion_.isValid()) {
      fileFormatVersion_.value_ = 0;
    }
    if (!fid_.isValid()) {
      fid_ = FileID(createGlobalIdentifier());
    }
    assert(eventTree_.isValid());
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
      if (eventAux_.luminosityBlock_ == 0 || fileFormatVersion_.value_ <= 1) {
        eventAux_.luminosityBlock_ = 1;
      }
    }
  }

  void
  RootFile::fillEventAuxiliaryAndHistory() {
    fillEventAuxiliary();
    if (!eventProcessHistoryIDs_.empty()) {
      if (eventProcessHistoryIter_->eventID_ != eventAux_.id()) {
        EventProcessHistoryID target(eventAux_.id(), ProcessHistoryID());
        eventProcessHistoryIter_ = std::lower_bound(eventProcessHistoryIDs_.begin(), eventProcessHistoryIDs_.end(), target);	
        assert(eventProcessHistoryIter_->eventID_ == eventAux_.id());
      }
      eventAux_.processHistoryID_ = eventProcessHistoryIter_->processHistoryID_;
      ++eventProcessHistoryIter_;
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
  RootFile::readEvent(boost::shared_ptr<ProductRegistry const> pReg, boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    assert(fileIndexIter_ != fileIndexEnd_);
    assert(fileIndexIter_->getEntryType() == FileIndex::kEvent);
    RunNumber_t currentRun = (fileIndexIter_->run_ ? fileIndexIter_->run_ : 1U);
    assert(currentRun >= startAtRun_);
    assert(currentRun > startAtRun_ || fileIndexIter_->lumi_ >= startAtLumi_);
    assert(currentRun > startAtRun_ || fileIndexIter_->lumi_ > startAtLumi_ ||
	 fileIndexIter_->event_ >= startAtEvent_);
    // Set the entry in the tree, and read the event at that entry.
    eventTree_.setEntryNumber(fileIndexIter_->entry_); 
    std::auto_ptr<EventPrincipal> ep = readCurrentEvent(pReg, lbp);

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
  RootFile::readCurrentEvent(boost::shared_ptr<ProductRegistry const> pReg, boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    if (!eventTree_.current()) {
      return std::auto_ptr<EventPrincipal>(0);
    }
    fillEventAuxiliaryAndHistory();
    overrideRunNumber(eventAux_.id_, eventAux_.isRealData());
    if (lbp.get() == 0) {
	boost::shared_ptr<RunPrincipal> rp(
	  new RunPrincipal(eventAux_.run(), eventAux_.time(), eventAux_.time(), pReg, processConfiguration_));
	lbp = boost::shared_ptr<LuminosityBlockPrincipal>(
	  new LuminosityBlockPrincipal(eventAux_.luminosityBlock(),
				       eventAux_.time(),
				       eventAux_.time(),
				       pReg,
				       rp,
				       processConfiguration_));
    }
    assert(eventAux_.run() == lbp->run());
    assert(eventAux_.luminosityBlock() == lbp->luminosityBlock());

    // We're not done ... so prepare the EventPrincipal
    std::auto_ptr<EventPrincipal> thisEvent(new EventPrincipal(
                eventID(),
		eventAux_.processGUID(),
		eventAux_.time(), pReg,
		lbp, processConfiguration_,
		eventAux_.isRealData(),
		eventAux_.experimentType(),
		eventAux_.bunchCrossing(),
                eventAux_.storeNumber(),
		eventAux_.processHistoryID_,
		eventTree_.makeDelayedReader()));

    // Create a group in the event for each product
    eventTree_.fillGroups(thisEvent->groupGetter());
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
      currentRun = (fileIndexIter_->run_ ? fileIndexIter_->run_ : 1U);
      if (currentRun == startAtRun_ && fileIndexIter_->lumi_ < startAtLumi_) {
        fileIndexIter_ = fileIndex_.findPosition(fileIndexIter_->run_, startAtLumi_, 0U);      
      }
      return boost::shared_ptr<RunPrincipal>(
          new RunPrincipal(run.run(),
	  eventAux_.time(),
	  Timestamp::invalidTimestamp(), pReg,
	  processConfiguration_));
    }
    runTree_.setEntryNumber(fileIndexIter_->entry_); 
    fillRunAuxiliary();
    assert(runAux_.run() == fileIndexIter_->run_);
    overrideRunNumber(runAux_.id_);
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
	new RunPrincipal(runAux_.run(),
			 runAux_.beginTime(),
			 runAux_.endTime(),
			 pReg,
			 processConfiguration_,
			 runAux_.processHistoryID_,
			 runTree_.makeDelayedReader()));
    // Create a group in the run for each product
    runTree_.fillGroups(thisRun->groupGetter());
    // Read in all the products now.
    thisRun->readImmediate();
    ++fileIndexIter_;
    currentRun = (fileIndexIter_->run_ ? fileIndexIter_->run_ : 1U);
    if (currentRun == startAtRun_ && fileIndexIter_->lumi_ < startAtLumi_) {
      fileIndexIter_ = fileIndex_.findPosition(fileIndexIter_->run_, startAtLumi_, 0U);      
    }
    return thisRun;
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  RootFile::readLumi(boost::shared_ptr<ProductRegistry const> pReg, boost::shared_ptr<RunPrincipal> rp) {
    assert(fileIndexIter_ != fileIndexEnd_);
    assert(fileIndexIter_->getEntryType() == FileIndex::kLumi);
    RunNumber_t currentRun = (fileIndexIter_->run_ ? fileIndexIter_->run_ : 1U);
    assert(currentRun >= startAtRun_);
    assert(currentRun > startAtRun_ || fileIndexIter_->lumi_ >= startAtLumi_);
    if (!lumiTree_.isValid()) {
        // prior to the support of lumi trees
      if (eventTree_.next()) {
        fillEventAuxiliary();
        // back up, so event will not be skipped.
        eventTree_.previous();
      }

      LuminosityBlockID lumi = LuminosityBlockID(fileIndexIter_->run_, fileIndexIter_->lumi_);
      overrideRunNumber(lumi);
      ++fileIndexIter_;
      currentRun = (fileIndexIter_->run_ ? fileIndexIter_->run_ : 1U);
      if (currentRun == startAtRun_ && fileIndexIter_->lumi_ == startAtLumi_ && fileIndexIter_->event_ < startAtEvent_) {
        fileIndexIter_ = fileIndex_.findPosition(fileIndexIter_->run_, startAtLumi_, startAtEvent_);      
      }
      while (eventsToSkip_ != 0 && fileIndexIter_ != fileIndexEnd_ &&
	   getEntryTypeSkippingDups() == FileIndex::kEvent) {
        ++fileIndexIter_;
        --eventsToSkip_;
      }
      return boost::shared_ptr<LuminosityBlockPrincipal>(
	new LuminosityBlockPrincipal(lumi.luminosityBlock(),
				     eventAux_.time_,
				     Timestamp::invalidTimestamp(),
				     pReg,
				     rp,
				     processConfiguration_));
    }
    lumiTree_.setEntryNumber(fileIndexIter_->entry_); 
    fillLumiAuxiliary();
    assert(lumiAux_.run() == fileIndexIter_->run_);
    assert(lumiAux_.luminosityBlock() == fileIndexIter_->lumi_);
    overrideRunNumber(lumiAux_.id_);
    assert(lumiAux_.run() == rp->run());

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
	new LuminosityBlockPrincipal(lumiAux_.luminosityBlock(),
				     lumiAux_.beginTime(),
				     lumiAux_.endTime(),
				     pReg, rp, processConfiguration_,
				     lumiAux_.processHistoryID_,
				     lumiTree_.makeDelayedReader()));
    // Create a group in the lumi for each product
    lumiTree_.fillGroups(thisLumi->groupGetter());
    // Read in all the products now.
    thisLumi->readImmediate();
    ++fileIndexIter_;
    currentRun = (fileIndexIter_->run_ ? fileIndexIter_->run_ : 1U);
    if (currentRun == startAtRun_ && fileIndexIter_->lumi_ == startAtLumi_ && fileIndexIter_->event_ < startAtEvent_) {
      fileIndexIter_ = fileIndex_.findPosition(fileIndexIter_->run_, startAtLumi_, startAtEvent_);      
    }
    while (eventsToSkip_ != 0 && fileIndexIter_ != fileIndexEnd_ &&
	 getEntryTypeSkippingDups() == FileIndex::kEvent) {
      ++fileIndexIter_;
      --eventsToSkip_;
    }
    return thisLumi;
  }


  
  bool
  RootFile::setEntryAtEvent(EventID const& id) {
    fileIndexIter_ = fileIndex_.findPosition(id.run(), 0U, id.event());
    while (fileIndexIter_ != fileIndexEnd_ && fileIndexIter_->getEntryType() != FileIndex::kEvent) {
      ++fileIndexIter_;
    }
    if (fileIndexIter_ == fileIndexEnd_) return false;
    eventTree_.setEntryNumber(fileIndexIter_->entry_);
    return true;
  }

  void
  RootFile::overrideRunNumber(RunID & id) {
    if (forcedRunOffset_ != 0) {
      id = RunID(id.run() + forcedRunOffset_);
    } 
    if (id.run() == 0) id = RunID::firstValidRun();
  }

  void
  RootFile::overrideRunNumber(LuminosityBlockID & id) {
    if (forcedRunOffset_ != 0) {
      id = LuminosityBlockID(id.run() + forcedRunOffset_, id.luminosityBlock());
    } 
    if (id.run() == 0) id = LuminosityBlockID(RunID::firstValidRun().run(), id.luminosityBlock());
  }

  void
  RootFile::overrideRunNumber(EventID & id, bool isRealData) {
    if (forcedRunOffset_ != 0) {
      if (isRealData) {
        throw cms::Exception("Configuration","RootFile::RootFile()")
          << "The 'setRunNumber' parameter of PoolSource cannot be used with real data.\n";
      }
      id = EventID(id.run() + forcedRunOffset_, id.event());
    } 
    if (id.run() == 0) id = EventID(RunID::firstValidRun().run(), id.event());
  }
}
