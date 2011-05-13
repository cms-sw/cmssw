
#include "IOPool/Output/src/RootOutputFile.h"

#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "FWCore/Version/interface/GetFileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/OutputHandle.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListRegistry.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "TClass.h"
#include "Rtypes.h"

#include <algorithm>
#include <iomanip>
#include <sstream>


namespace edm {

  namespace {
    bool
    sorterForJobReportHash(BranchDescription const* lh, BranchDescription const* rh) {
      return
        lh->fullClassName() < rh->fullClassName() ? true :
        lh->fullClassName() > rh->fullClassName() ? false :
        lh->moduleLabel() < rh->moduleLabel() ? true :
        lh->moduleLabel() > rh->moduleLabel() ? false :
        lh->productInstanceName() < rh->productInstanceName() ? true :
        lh->productInstanceName() > rh->productInstanceName() ? false :
        lh->processName() < rh->processName() ? true :
        false;
    }
  }

  RootOutputFile::RootOutputFile(PoolOutputModule* om, std::string const& fileName, std::string const& logicalFileName) :
      file_(fileName),
      logicalFile_(logicalFileName),
      reportToken_(0),
      om_(om),
      whyNotFastClonable_(om_->whyNotFastClonable()),
      canFastCloneAux_(false),
      filePtr_(TFile::Open(file_.c_str(), "recreate", "", om_->compressionLevel())),
      fid_(),
      eventEntryNumber_(0LL),
      lumiEntryNumber_(0LL),
      runEntryNumber_(0LL),
      indexIntoFile_(),
      metaDataTree_(0),
      parameterSetsTree_(0),
      parentageTree_(0),
      lumiAux_(),
      runAux_(),
      pEventAux_(0),
      pLumiAux_(&lumiAux_),
      pRunAux_(&runAux_),
      eventEntryInfoVector_(),
      lumiEntryInfoVector_(),
      runEntryInfoVector_(),
      pEventEntryInfoVector_(&eventEntryInfoVector_),
      pLumiEntryInfoVector_(&lumiEntryInfoVector_),
      pRunEntryInfoVector_(&runEntryInfoVector_),
      pBranchListIndexes_(0),
      pEventSelectionIDs_(0),
      eventTree_(filePtr_, InEvent, om_->splitLevel(), om_->treeMaxVirtualSize()),
      lumiTree_(filePtr_, InLumi, om_->splitLevel(), om_->treeMaxVirtualSize()),
      runTree_(filePtr_, InRun, om_->splitLevel(), om_->treeMaxVirtualSize()),
      treePointers_(),
      dataTypeReported_(false),
      parentageIDs_(),
      branchesWithStoredHistory_() {

    if (-1 != om->eventAutoFlushSize()) {
      eventTree_.setAutoFlush(-1*om->eventAutoFlushSize());
    }
    eventTree_.addAuxiliary<EventAuxiliary>(BranchTypeToAuxiliaryBranchName(InEvent),
                                            pEventAux_, om_->auxItems()[InEvent].basketSize_);
    eventTree_.addAuxiliary<ProductProvenanceVector>(BranchTypeToBranchEntryInfoBranchName(InEvent),
                                                     pEventEntryInfoVector_, om_->auxItems()[InEvent].basketSize_);
    eventTree_.addAuxiliary<EventSelectionIDVector>(poolNames::eventSelectionsBranchName(),
                                                    pEventSelectionIDs_, om_->auxItems()[InEvent].basketSize_);
    eventTree_.addAuxiliary<BranchListIndexes>(poolNames::branchListIndexesBranchName(),
                                               pBranchListIndexes_, om_->auxItems()[InEvent].basketSize_);

    lumiTree_.addAuxiliary<LuminosityBlockAuxiliary>(BranchTypeToAuxiliaryBranchName(InLumi),
                                                     pLumiAux_, om_->auxItems()[InLumi].basketSize_);
    lumiTree_.addAuxiliary<ProductProvenanceVector>(BranchTypeToBranchEntryInfoBranchName(InLumi),
                                                    pLumiEntryInfoVector_, om_->auxItems()[InLumi].basketSize_);

    runTree_.addAuxiliary<RunAuxiliary>(BranchTypeToAuxiliaryBranchName(InRun),
                                        pRunAux_, om_->auxItems()[InRun].basketSize_);
    runTree_.addAuxiliary<ProductProvenanceVector>(BranchTypeToBranchEntryInfoBranchName(InRun),
                                                   pRunEntryInfoVector_, om_->auxItems()[InRun].basketSize_);

    treePointers_[InEvent] = &eventTree_;
    treePointers_[InLumi]  = &lumiTree_;
    treePointers_[InRun]   = &runTree_;

    for(int i = InEvent; i < NumBranchTypes; ++i) {
      BranchType branchType = static_cast<BranchType>(i);
      RootOutputTree *theTree = treePointers_[branchType];
      for(OutputItemList::const_iterator it = om_->selectedOutputItemList()[branchType].begin(),
          itEnd = om_->selectedOutputItemList()[branchType].end();
          it != itEnd; ++it) {
        it->product_ = 0;
        BranchDescription const& desc = *it->branchDescription_;
        desc.init();
        theTree->addBranch(desc.branchName(),
                           desc.wrappedName(),
                           desc.getInterface(),
                           it->product_,
                           it->splitLevel_,
                           it->basketSize_,
                           it->branchDescription_->produced());
        //make sure we always store product registry info for all branches we create
        branchesWithStoredHistory_.insert(it->branchID());
      }
    }
    // Don't split metadata tree or event description tree
    metaDataTree_         = RootOutputTree::makeTTree(filePtr_.get(), poolNames::metaDataTreeName(), 0);
    parentageTree_ = RootOutputTree::makeTTree(filePtr_.get(), poolNames::parentageTreeName(), 0);
    parameterSetsTree_    = RootOutputTree::makeTTree(filePtr_.get(), poolNames::parameterSetsTreeName(), 0);

    fid_ = FileID(createGlobalIdentifier());

    // For the Job Report, get a vector of branch names in the "Events" tree.
    // Also create a hash of all the branch names in the "Events" tree
    // in a deterministic order, except use the full class name instead of the friendly class name.
    // To avoid extra string copies, we create a vector of pointers into the product registry,
    // and use a custom comparison operator for sorting.
    std::vector<std::string> branchNames;
    std::vector<BranchDescription const*> branches;
    branchNames.reserve(om_->selectedOutputItemList()[InEvent].size());
    branches.reserve(om->selectedOutputItemList()[InEvent].size());
    for(OutputItemList::const_iterator it = om_->selectedOutputItemList()[InEvent].begin(),
          itEnd = om_->selectedOutputItemList()[InEvent].end();
          it != itEnd; ++it) {
      branchNames.push_back(it->branchDescription_->branchName());
      branches.push_back(it->branchDescription_);
    }
    // Now sort the branches for the hash.
    sort_all(branches, sorterForJobReportHash);
    // Now, make a concatenated string.
    std::ostringstream oss;
    char const underscore = '_';
    for(std::vector<BranchDescription const*>::const_iterator it = branches.begin(), itEnd = branches.end(); it != itEnd; ++it) {
      BranchDescription const& bd = **it;
      oss << bd.fullClassName() << underscore
          << bd.moduleLabel() << underscore
          << bd.productInstanceName() << underscore
          << bd.processName() << underscore;
    }
    std::string stringrep = oss.str();
    cms::Digest md5alg(stringrep);

    // Register the output file with the JobReport service
    // and get back the token for it.
    std::string moduleName = "PoolOutputModule";
    Service<JobReport> reportSvc;
    reportToken_ = reportSvc->outputFileOpened(
                      file_, logicalFile_,  // PFN and LFN
                      om_->catalog(),  // catalog
                      moduleName,   // module class name
                      om_->moduleLabel(),  // module label
                      fid_.fid(), // file id (guid)
                      std::string(), // data type (not yet known, so string is empty).
                      md5alg.digest().toString(), // branch hash
                      branchNames); // branch names being written
  }

  namespace {
    void
    maybeIssueWarning(int whyNotFastClonable, std::string const& ifileName, std::string const& ofileName) {

      // No message if fast cloning was deliberately disabled, or if there are no events to copy anyway.
      if ((whyNotFastClonable &
        (FileBlock::DisabledInConfigFile | FileBlock::NoRootInputSource | FileBlock::NotProcessingEvents | FileBlock::NoEventsInFile)) != 0) {
        return;
      }

      // There will be a message stating every reason that fast cloning was not possible.
      // If at one or more of the reasons was because of something the user explicitly specified (e.g. event selection, skipping events),
      // or if the input file was in an old format, the message will be informational.  Otherwise, the message will be a warning.
      bool isWarning = true;
      std::ostringstream message;
      message << "Fast copying of file " << ifileName << " to file " << ofileName << " is disabled because:\n";
      if((whyNotFastClonable & FileBlock::HasSecondaryFileSequence) != 0) {
        message << "a SecondaryFileSequence was specified.\n";
        whyNotFastClonable &= ~(FileBlock::HasSecondaryFileSequence);
        isWarning = false;
      }
      if((whyNotFastClonable & FileBlock::FileTooOld) != 0) {
        message << "the input file is in an old format.\n";
        whyNotFastClonable &= ~(FileBlock::FileTooOld);
        isWarning = false;
      }
      if((whyNotFastClonable & FileBlock::EventsToBeSorted) != 0) {
        message << "events need to be sorted.\n";
        whyNotFastClonable &= ~(FileBlock::EventsToBeSorted);
      }
      if((whyNotFastClonable & FileBlock::EventsOrLumisSelectedByID) != 0) {
        message << "events or lumis were selected or skipped by ID.\n";
        whyNotFastClonable &= ~(FileBlock::EventsOrLumisSelectedByID);
        isWarning = false;
      }
      if((whyNotFastClonable & FileBlock::InitialEventsSkipped) != 0) {
        message << "initial events, lumis or runs were skipped.\n";
        whyNotFastClonable &= ~(FileBlock::InitialEventsSkipped);
        isWarning = false;
      }
      if((whyNotFastClonable & FileBlock::DuplicateEventsRemoved) != 0) {
        message << "some events were skipped because of duplicate checking.\n";
        whyNotFastClonable &= ~(FileBlock::DuplicateEventsRemoved);
      }
      if((whyNotFastClonable & FileBlock::MaxEventsTooSmall) != 0) {
        message << "some events were not copied because of maxEvents limit.\n";
        whyNotFastClonable &= ~(FileBlock::MaxEventsTooSmall);
        isWarning = false;
      }
      if((whyNotFastClonable & FileBlock::MaxLumisTooSmall) != 0) {
        message << "some events were not copied because of maxLumis limit.\n";
        whyNotFastClonable &= ~(FileBlock::MaxLumisTooSmall);
        isWarning = false;
      }
      if((whyNotFastClonable & FileBlock::ParallelProcesses) != 0) {
        message << "parallel processing was specified.\n";
        whyNotFastClonable &= ~(FileBlock::ParallelProcesses);
        isWarning = false;
      }
      if((whyNotFastClonable & FileBlock::EventSelectionUsed) != 0) {
        message << "an EventSelector was specified.\n";
        whyNotFastClonable &= ~(FileBlock::EventSelectionUsed);
        isWarning = false;
      }
      if((whyNotFastClonable & FileBlock::OutputMaxEventsTooSmall) != 0) {
        message << "some events were not copied because of maxEvents output limit.\n";
        whyNotFastClonable &= ~(FileBlock::OutputMaxEventsTooSmall);
        isWarning = false;
      }
      if((whyNotFastClonable & FileBlock::SplitLevelMismatch) != 0) {
        message << "the split level or basket size of a branch or branches was modified.\n";
        whyNotFastClonable &= ~(FileBlock::SplitLevelMismatch);
      }
      if((whyNotFastClonable & FileBlock::BranchMismatch) != 0) {
        message << "The format of a data product has changed.\n";
        whyNotFastClonable &= ~(FileBlock::BranchMismatch);
      }
      assert(whyNotFastClonable == FileBlock::CanFastClone);
      if (isWarning) {
        LogWarning("FastCloningDisabled") << message.str();
      } else {
        LogInfo("FastCloningDisabled") << message.str();
      }
    }
  }

  void RootOutputFile::beginInputFile(FileBlock const& fb, int remainingEvents) {

    // Reset per input file information
    whyNotFastClonable_ = om_->whyNotFastClonable();
    canFastCloneAux_ = false;

    if(fb.tree() != 0) {

      whyNotFastClonable_ |= fb.whyNotFastClonable();

      if(remainingEvents >= 0 && remainingEvents < fb.tree()->GetEntries()) {
        whyNotFastClonable_ |= FileBlock::OutputMaxEventsTooSmall;
      }

      bool match = eventTree_.checkSplitLevelsAndBasketSizes(fb.tree());
      if(!match) {
        if(om_->overrideInputFileSplitLevels()) {
          // We may be fast copying.  We must disable fast copying if the split levels
          // or basket sizes do not match.
          whyNotFastClonable_ |= FileBlock::SplitLevelMismatch;
        } else {
          // We are using the input split levels and basket sizes from the first input file
          // for copied output branches.  In this case, we throw an exception if any branches
          // have different split levels or basket sizes in a subsequent input file.
          // If the mismatch is in the first file, there is a bug somewhere, so we assert.
          assert(om_->inputFileCount() > 1);
          throw Exception(errors::MismatchedInputFiles, "RootOutputFile::beginInputFile()") <<
            "Merge failure because input file " << file_ << " has different ROOT split levels or basket sizes\n" <<
            "than previous files.  To allow merging in splite of this, use the configuration parameter\n" <<
            "overrideInputFileSplitLevels=cms.untracked.bool(True)\n" <<
            "in every PoolOutputModule.\n";
        }
      }

      // Since this check can be time consuming, we do it only if we would otherwise fast clone.
      if(whyNotFastClonable_ == FileBlock::CanFastClone) {
        if(!eventTree_.checkIfFastClonable(fb.tree())) {
          whyNotFastClonable_ |= FileBlock::BranchMismatch;
        }
      }
      // We now check if we can fast copy the auxiliary branches.
      // We can do so only if we can otherwise fast copy,
      // the input file has the current format (these branches are in the Events Tree),
      // there are no newly dropped or produced products,
      // and the branch list indexes do not need modification.
      Service<ConstProductRegistry> reg;
      canFastCloneAux_ = (whyNotFastClonable_ == FileBlock::CanFastClone) &&
                          fb.fileFormatVersion().noMetaDataTrees() &&
                          !om_->hasNewlyDroppedBranch()[InEvent] &&
                          !fb.hasNewlyDroppedBranch()[InEvent] &&
                          om_->dropMetaData() == PoolOutputModule::DropNone &&
                          !reg->anyProductProduced() &&
                          fb.branchListIndexesUnchanged();

      // Report the fast copying status.
      Service<JobReport> reportSvc;
      reportSvc->reportFastCopyingStatus(reportToken_, fb.fileName(), whyNotFastClonable_ == FileBlock::CanFastClone);
    } else {
      whyNotFastClonable_ |= FileBlock::NoRootInputSource;
    }

    eventTree_.maybeFastCloneTree(whyNotFastClonable_ == FileBlock::CanFastClone, canFastCloneAux_, fb.tree(), om_->basketOrder());

    // Possibly issue warning or informational message if we haven't fast cloned.
    if(fb.tree() != 0 && whyNotFastClonable_ != FileBlock::CanFastClone) {
      maybeIssueWarning(whyNotFastClonable_, fb.fileName(), file_);
    }
  }

  void RootOutputFile::respondToCloseInputFile(FileBlock const&) {
    eventTree_.setEntries();
    lumiTree_.setEntries();
    runTree_.setEntries();
  }

  bool RootOutputFile::shouldWeCloseFile() const {
    unsigned int const oneK = 1024;
    Long64_t size = filePtr_->GetSize()/oneK;
    return(size >= om_->maxFileSize());
  }

  void RootOutputFile::writeOne(EventPrincipal const& e) {
    // Auxiliary branch
    pEventAux_ = &e.aux();

    // Because getting the data may cause an exception to be thrown we want to do that
    // first before writing anything to the file about this event
    // NOTE: pEventAux_, pBranchListIndexes_, pEventSelectionIDs_, and pEventEntryInfoVector_
    // must be set before calling fillBranches since they get written out in that routine.
    assert(pEventAux_->processHistoryID() == e.processHistoryID());
    pBranchListIndexes_ = &e.branchListIndexes();

    // Note: The EventSelectionIDVector should have a one to one correspondence with the processes in the process history.
    // Therefore, a new entry should be added if and only if the current process has been added to the process history,
    // which is done if and only if there is a produced product.
    Service<ConstProductRegistry> reg;
    EventSelectionIDVector esids = e.eventSelectionIDs();
    if (reg->anyProductProduced() || !om_->wantAllEvents()) {
      esids.push_back(om_->selectorConfig());
    }
    pEventSelectionIDs_ = &esids;
    fillBranches(InEvent, e, pEventEntryInfoVector_);

    // Add the dataType to the job report if it hasn't already been done
    if(!dataTypeReported_) {
      Service<JobReport> reportSvc;
      std::string dataType("MC");
      if(pEventAux_->isRealData())  dataType = "Data";
      reportSvc->reportDataType(reportToken_, dataType);
      dataTypeReported_ = true;
    }

    // Add event to index
    indexIntoFile_.addEntry(e.processHistoryID(), pEventAux_->run(), pEventAux_->luminosityBlock(), pEventAux_->event(), eventEntryNumber_);
    ++eventEntryNumber_;

    // Report event written
    Service<JobReport> reportSvc;
    reportSvc->eventWrittenToFile(reportToken_, e.id().run(), e.id().event());
  }

  void RootOutputFile::writeLuminosityBlock(LuminosityBlockPrincipal const& lb) {
    // Auxiliary branch
    // NOTE: lumiAux_ must be filled before calling fillBranches since it gets written out in that routine.
    lumiAux_ = lb.aux();
    // Use the updated process historyID
    lumiAux_.setProcessHistoryID(lb.processHistoryID());
    // Add lumi to index.
    indexIntoFile_.addEntry(lb.processHistoryID(), lumiAux_.run(), lumiAux_.luminosityBlock(), 0U, lumiEntryNumber_);
    ++lumiEntryNumber_;
    fillBranches(InLumi, lb, pLumiEntryInfoVector_);
    lumiTree_.optimizeBaskets(10ULL*1024*1024);
  }

  void RootOutputFile::writeRun(RunPrincipal const& r) {
    // Auxiliary branch
    // NOTE: runAux_ must be filled before calling fillBranches since it gets written out in that routine.
    runAux_ = r.aux();
    // Use the updated process historyID
    runAux_.setProcessHistoryID(r.processHistoryID());
    // Add run to index.
    indexIntoFile_.addEntry(r.processHistoryID(), runAux_.run(), 0U, 0U, runEntryNumber_);
    ++runEntryNumber_;
    fillBranches(InRun, r, pRunEntryInfoVector_);
    runTree_.optimizeBaskets(10ULL*1024*1024);
  }

  void RootOutputFile::writeParentageRegistry() {
    Parentage const* desc(0);

    if(!parentageTree_->Branch(poolNames::parentageBranchName().c_str(),
                                        &desc, om_->basketSize(), 0))
      throw Exception(errors::FatalRootError)
        << "Failed to create a branch for Parentages in the output file";

    ParentageRegistry& ptReg = *ParentageRegistry::instance();
    std::set<ParentageID>::const_iterator pidend = parentageIDs_.end();
    for(ParentageRegistry::const_iterator i = ptReg.begin(), e = ptReg.end(); i != e; ++i) {
      if(parentageIDs_.find(i->first) != pidend) {
        desc = &(i->second);
        parentageTree_->Fill();
      }
    }
  }

  void RootOutputFile::writeFileFormatVersion() {
    FileFormatVersion fileFormatVersion(getFileFormatVersion());
    FileFormatVersion* pFileFmtVsn = &fileFormatVersion;
    TBranch* b = metaDataTree_->Branch(poolNames::fileFormatVersionBranchName().c_str(), &pFileFmtVsn, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeFileIdentifier() {
    FileID* fidPtr = &fid_;
    TBranch* b = metaDataTree_->Branch(poolNames::fileIdentifierBranchName().c_str(), &fidPtr, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeIndexIntoFile() {
    if (eventTree_.checkEntriesInReadBranches(eventEntryNumber_) == false) {
      Exception ex(errors::OtherCMS);
      ex << "The number of entries in at least one output TBranch whose entries\n"
            "were copied from the input does not match the number of events\n"
            "recorded in IndexIntoFile. This might (or might not) indicate a\n"
            "problem related to fast copy.";
      ex.addContext("Calling RootOutputFile::writeIndexIntoFile");
      throw ex;
    }
    indexIntoFile_.sortVector_Run_Or_Lumi_Entries();
    IndexIntoFile* iifPtr = &indexIntoFile_;
    TBranch* b = metaDataTree_->Branch(poolNames::indexIntoFileBranchName().c_str(), &iifPtr, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeProcessConfigurationRegistry() {
    typedef ProcessConfigurationRegistry::collection_type Map;
    Map const& procConfigMap = ProcessConfigurationRegistry::instance()->data();
    ProcessConfigurationVector procConfigVector;
    for(Map::const_iterator i = procConfigMap.begin(), e = procConfigMap.end(); i != e; ++i) {
      procConfigVector.push_back(i->second);
    }
    sort_all(procConfigVector);
    ProcessConfigurationVector* p = &procConfigVector;
    TBranch* b = metaDataTree_->Branch(poolNames::processConfigurationBranchName().c_str(), &p, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeProcessHistoryRegistry() {
    typedef ProcessHistoryRegistry::collection_type Map;
    Map const& procHistoryMap = ProcessHistoryRegistry::instance()->data();
    ProcessHistoryVector procHistoryVector;
    for(Map::const_iterator i = procHistoryMap.begin(), e = procHistoryMap.end(); i != e; ++i) {
      procHistoryVector.push_back(i->second);
    }
    ProcessHistoryVector* p = &procHistoryVector;
    TBranch* b = metaDataTree_->Branch(poolNames::processHistoryBranchName().c_str(), &p, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeBranchIDListRegistry() {
    BranchIDListRegistry::collection_type* p = &BranchIDListRegistry::instance()->data();
    TBranch* b = metaDataTree_->Branch(poolNames::branchIDListBranchName().c_str(), &p, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeParameterSetRegistry() {
    std::pair<ParameterSetID, ParameterSetBlob> idToBlob;
    std::pair<ParameterSetID, ParameterSetBlob>* pIdToBlob = &idToBlob;
    TBranch* b = parameterSetsTree_->Branch(poolNames::idToParameterSetBlobsBranchName().c_str(),&pIdToBlob,om_->basketSize(), 0);

    for(pset::Registry::const_iterator it = pset::Registry::instance()->begin(),
        itEnd = pset::Registry::instance()->end();
        it != itEnd;
        ++it) {
      idToBlob.first = it->first;
      idToBlob.second.pset() = it->second.toString();

      b->Fill();
    }
  }

  void RootOutputFile::writeProductDescriptionRegistry() {
    // Make a local copy of the ProductRegistry, removing any transient or pruned products.
    typedef ProductRegistry::ProductList ProductList;
    Service<ConstProductRegistry> reg;
    ProductRegistry pReg(reg->productList());
    ProductList& pList  = const_cast<ProductList &>(pReg.productList());
    std::set<BranchID>::iterator end = branchesWithStoredHistory_.end();
    for(ProductList::iterator it = pList.begin(); it != pList.end();) {
      if(branchesWithStoredHistory_.find(it->second.branchID()) == end) {
        // avoid invalidating iterator on deletion
        ProductList::iterator itCopy = it;
        ++it;
        pList.erase(itCopy);

      } else {
        ++it;
      }
    }

    ProductRegistry* ppReg = &pReg;
    TBranch* b = metaDataTree_->Branch(poolNames::productDescriptionBranchName().c_str(), &ppReg, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }
  void RootOutputFile::writeProductDependencies() {
    BranchChildren& pDeps = const_cast<BranchChildren&>(om_->branchChildren());
    BranchChildren* ppDeps = &pDeps;
    TBranch* b = metaDataTree_->Branch(poolNames::productDependenciesBranchName().c_str(), &ppDeps, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::finishEndFile() {
    metaDataTree_->SetEntries(-1);
    RootOutputTree::writeTTree(metaDataTree_);
    RootOutputTree::writeTTree(parameterSetsTree_);

    RootOutputTree::writeTTree(parentageTree_);

    // Create branch aliases for all the branches in the
    // events/lumis/runs trees. The loop is over all types of data
    // products.
    for(int i = InEvent; i < NumBranchTypes; ++i) {
      BranchType branchType = static_cast<BranchType>(i);
      setBranchAliases(treePointers_[branchType]->tree(), om_->keptProducts()[branchType]);
      treePointers_[branchType]->writeTree();
    }

    // close the file -- mfp
    // Just to play it safe, zero all pointers to objects in the TFile to be closed.
    metaDataTree_ = parentageTree_ = 0;
    for(RootOutputTreePtrArray::iterator it = treePointers_.begin(), itEnd = treePointers_.end(); it != itEnd; ++it) {
      (*it)->close();
      (*it) = 0;
    }
    filePtr_->Close();
    filePtr_.reset();

    // report that file has been closed
    Service<JobReport> reportSvc;
    reportSvc->outputFileClosed(reportToken_);

  }

  void
  RootOutputFile::setBranchAliases(TTree* tree, Selections const& branches) const {
    if(tree && tree->GetNbranches() != 0) {
      for(Selections::const_iterator i = branches.begin(), iEnd = branches.end();
          i != iEnd; ++i) {
        BranchDescription const& pd = **i;
        std::string const& full = pd.branchName() + "obj";
        if(pd.branchAliases().empty()) {
          std::string const& alias =
              (pd.productInstanceName().empty() ? pd.moduleLabel() : pd.productInstanceName());
          tree->SetAlias(alias.c_str(), full.c_str());
        } else {
          std::set<std::string>::const_iterator it = pd.branchAliases().begin(), itEnd = pd.branchAliases().end();
          for(; it != itEnd; ++it) {
            tree->SetAlias((*it).c_str(), full.c_str());
          }
        }
      }
    }
  }

  void
  RootOutputFile::insertAncestors(ProductProvenance const& iGetParents,
                                  Principal const& principal,
                                  bool produced,
                                  std::set<ProductProvenance>& oToFill) {
    assert(om_->dropMetaData() != PoolOutputModule::DropAll);
    assert(produced || om_->dropMetaData() != PoolOutputModule::DropPrior);
    if(om_->dropMetaData() == PoolOutputModule::DropDroppedPrior && !produced) return;
    BranchMapper const& iMapper = *principal.branchMapperPtr();
    std::vector<BranchID> const& parentIDs = iGetParents.parentage().parents();
    for(std::vector<BranchID>::const_iterator it = parentIDs.begin(), itEnd = parentIDs.end();
          it != itEnd; ++it) {
      branchesWithStoredHistory_.insert(*it);
      boost::shared_ptr<ProductProvenance> info = iMapper.branchIDToProvenance(*it);
      if(info) {
        if(om_->dropMetaData() == PoolOutputModule::DropNone ||
                 principal.getProvenance(info->branchID()).product().produced()) {
          if(oToFill.insert(*info).second) {
            //haven't seen this one yet
            insertAncestors(*info, principal, produced, oToFill);
          }
        }
      }
    }
  }

  void RootOutputFile::fillBranches(
                BranchType const& branchType,
                Principal const& principal,
                ProductProvenanceVector* productProvenanceVecPtr) {

    typedef std::vector<std::pair<TClass*, void const*> > Dummies;
    Dummies dummies;

    bool const fastCloning = (branchType == InEvent) && (whyNotFastClonable_ == FileBlock::CanFastClone);

    OutputItemList const& items = om_->selectedOutputItemList()[branchType];

    std::set<ProductProvenance> provenanceToKeep;

    // Loop over EDProduct branches, fill the provenance, and write the branch.
    for(OutputItemList::const_iterator i = items.begin(), iEnd = items.end(); i != iEnd; ++i) {

      BranchID const& id = i->branchDescription_->branchID();
      branchesWithStoredHistory_.insert(id);

      bool produced = i->branchDescription_->produced();
      bool keepProvenance = om_->dropMetaData() == PoolOutputModule::DropNone ||
                            om_->dropMetaData() == PoolOutputModule::DropDroppedPrior ||
                           (om_->dropMetaData() == PoolOutputModule::DropPrior && produced);
      bool getProd = (produced || !fastCloning ||
         treePointers_[branchType]->uncloned(i->branchDescription_->branchName()));

      void const* product = 0;
      OutputHandle const oh = principal.getForOutput(id, getProd);
      if(keepProvenance && oh.productProvenance()) {
        provenanceToKeep.insert(*oh.productProvenance());
        assert(principal.branchMapperPtr());
        insertAncestors(*oh.productProvenance(), principal, produced, provenanceToKeep);
      }
      product = oh.wrapper();
      if(getProd) {
        if(product == 0) {
          // No product with this ID is in the event.
          // Add a null product.
          TClass* cp = gROOT->GetClass(i->branchDescription_->wrappedName().c_str());
          product = cp->New();
          dummies.push_back(std::make_pair(cp, product));
        }
        i->product_ = product;
      }
    }

    for(std::set<ProductProvenance>::const_iterator it = provenanceToKeep.begin(), itEnd=provenanceToKeep.end();
        it != itEnd; ++it) {
      parentageIDs_.insert(it->parentageID());
    }

    productProvenanceVecPtr->assign(provenanceToKeep.begin(), provenanceToKeep.end());
    treePointers_[branchType]->fillTree();
    productProvenanceVecPtr->clear();
    for(Dummies::iterator it = dummies.begin(), itEnd = dummies.end(); it != itEnd; ++it) {
      it->first->Destructor(const_cast<void *>(it->second));
    }
  }
}
