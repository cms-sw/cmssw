/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootFile.h"
#include "DuplicateChecker.h"
#include "InputFile.h"
#include "ProvenanceAdaptor.h"

#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/FullHistoryToReducedHistoryMap.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/StoredProductProvenance.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FillProductRegistryTransients.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"
#include "FWCore/Sources/interface/DaqProvenanceHelper.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/ReleaseVersion.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

//used for backward compatibility
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/EventAux.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"
#include "DataFormats/Provenance/interface/RunAux.h"
#include "FWCore/ParameterSet/interface/ParameterSetConverter.h"

#include "TROOT.h"
#include "Rtypes.h"
#include "TClass.h"
#include "TString.h"
#include "TTree.h"
#include "TTreeCache.h"

#include <algorithm>
#include <list>

namespace edm {

  // Algorithm classes for making ProvenanceReader:
  class MakeDummyProvenanceReader : public MakeProvenanceReader {
  public:
    virtual std::unique_ptr<ProvenanceReaderBase> makeReader(RootTree& eventTree, DaqProvenanceHelper const* daqProvenanceHelper) const;
  };
  class MakeOldProvenanceReader : public MakeProvenanceReader {
  public:
    virtual std::unique_ptr<ProvenanceReaderBase> makeReader(RootTree& eventTree, DaqProvenanceHelper const* daqProvenanceHelper) const;
  };
  class MakeFullProvenanceReader : public MakeProvenanceReader {
  public:
    virtual std::unique_ptr<ProvenanceReaderBase> makeReader(RootTree& eventTree, DaqProvenanceHelper const* daqProvenanceHelper) const;
  };
  class MakeReducedProvenanceReader : public MakeProvenanceReader {
  public:
    MakeReducedProvenanceReader(std::vector<ParentageID> const& parentageIDLookup) : parentageIDLookup_(parentageIDLookup) {}
    virtual std::unique_ptr<ProvenanceReaderBase> makeReader(RootTree& eventTree, DaqProvenanceHelper const* daqProvenanceHelper) const;
  private:
    std::vector<ParentageID> const& parentageIDLookup_;
  };

  namespace {
    int
    forcedRunOffset(RunNumber_t const& forcedRunNumber, IndexIntoFile::IndexIntoFileItr inxBegin, IndexIntoFile::IndexIntoFileItr inxEnd) {
      if(inxBegin == inxEnd) return 0;
      int defaultOffset = (inxBegin.run() != 0 ? 0 : 1);
      int offset = (forcedRunNumber != 0U ? forcedRunNumber - inxBegin.run() : defaultOffset);
      if(offset < 0) {
        throw Exception(errors::Configuration)
          << "The value of the 'setRunNumber' parameter must not be\n"
          << "less than the first run number in the first input file.\n"
          << "'setRunNumber' was " << forcedRunNumber <<", while the first run was "
          << forcedRunNumber - offset << ".\n";
      }
      return offset;
    }
  }

  // This is a helper class for IndexIntoFile.
  class RootFileEventFinder : public IndexIntoFile::EventFinder {
  public:
    explicit RootFileEventFinder(RootTree& eventTree) : eventTree_(eventTree) {}
    virtual ~RootFileEventFinder() {}
    virtual
    EventNumber_t getEventNumberOfEntry(roottree::EntryNumber entry) const {
      roottree::EntryNumber saveEntry = eventTree_.entryNumber();
      eventTree_.setEntryNumber(entry);
      EventAuxiliary eventAux;
      EventAuxiliary *pEvAux = &eventAux;
      eventTree_.fillAux<EventAuxiliary>(pEvAux);
      eventTree_.setEntryNumber(saveEntry);
      return eventAux.event();
    }

  private:
     RootTree& eventTree_;
  };

//---------------------------------------------------------------------
  RootFile::RootFile(std::string const& fileName,
                     ProcessConfiguration const& processConfiguration,
                     std::string const& logicalFileName,
                     boost::shared_ptr<InputFile> filePtr,
                     boost::shared_ptr<EventSkipperByID> eventSkipperByID,
                     bool skipAnyEvents,
                     int remainingEvents,
                     int remainingLumis,
                     unsigned int treeCacheSize,
                     int treeMaxVirtualSize,
                     InputSource::ProcessingMode processingMode,
                     RunNumber_t const& forcedRunNumber,
                     bool noEventSort,
                     ProductSelectorRules const& productSelectorRules,
                     InputType::InputType inputType,
                     boost::shared_ptr<BranchIDListHelper> branchIDListHelper,
                     boost::shared_ptr<DuplicateChecker> duplicateChecker,
                     bool dropDescendants,
                     std::vector<boost::shared_ptr<IndexIntoFile> > const& indexesIntoFiles,
                     std::vector<boost::shared_ptr<IndexIntoFile> >::size_type currentIndexIntoFile,
                     std::vector<ProcessHistoryID>& orderedProcessHistoryIDs,
                     bool labelRawDataLikeMC,
                     bool usingGoToEvent,
                     bool enablePrefetching) :
      file_(fileName),
      logicalFile_(logicalFileName),
      processConfiguration_(processConfiguration),
      processConfigurations_(),
      filePtr_(filePtr),
      eventSkipperByID_(eventSkipperByID),
      fileFormatVersion_(),
      fid_(),
      indexIntoFileSharedPtr_(new IndexIntoFile),
      indexIntoFile_(*indexIntoFileSharedPtr_),
      orderedProcessHistoryIDs_(orderedProcessHistoryIDs),
      indexIntoFileBegin_(indexIntoFile_.begin(noEventSort ? IndexIntoFile::firstAppearanceOrder : IndexIntoFile::numericalOrder)),
      indexIntoFileEnd_(indexIntoFileBegin_),
      indexIntoFileIter_(indexIntoFileBegin_),
      eventProcessHistoryIDs_(),
      eventProcessHistoryIter_(eventProcessHistoryIDs_.begin()),
      savedRunAuxiliary_(),
      skipAnyEvents_(skipAnyEvents),
      noEventSort_(noEventSort),
      whyNotFastClonable_(0),
      hasNewlyDroppedBranch_(),
      branchListIndexesUnchanged_(false),
      eventAux_(),
      eventTree_(filePtr_, InEvent, treeMaxVirtualSize, treeCacheSize, roottree::defaultLearningEntries, enablePrefetching),
      lumiTree_(filePtr_, InLumi, treeMaxVirtualSize, roottree::defaultNonEventCacheSize, roottree::defaultNonEventLearningEntries, enablePrefetching),
      runTree_(filePtr_, InRun, treeMaxVirtualSize, roottree::defaultNonEventCacheSize, roottree::defaultNonEventLearningEntries, enablePrefetching),
      treePointers_(),
      lastEventEntryNumberRead_(-1LL),
      productRegistry_(),
      branchIDLists_(),
      branchIDListHelper_(branchIDListHelper),
      processingMode_(processingMode),
      forcedRunOffset_(0),
      newBranchToOldBranch_(),
      eventHistoryTree_(nullptr),
      eventSelectionIDs_(new EventSelectionIDVector),
      branchListIndexes_(new BranchListIndexes),
      history_(),
      branchChildren_(new BranchChildren),
      duplicateChecker_(duplicateChecker),
      provenanceAdaptor_(),
      provenanceReaderMaker_(),
      secondaryEventPrincipal_(),
      eventBranchMapper_(),
      parentageIDLookup_(),
      daqProvenanceHelper_() {

    hasNewlyDroppedBranch_.fill(false);

    treePointers_[InEvent] = &eventTree_;
    treePointers_[InLumi]  = &lumiTree_;
    treePointers_[InRun]   = &runTree_;

    // Read the metadata tree.
    // We use a smart pointer so the tree will be deleted after use, and not kept for the life of the file.
    std::unique_ptr<TTree> metaDataTree(dynamic_cast<TTree *>(filePtr_->Get(poolNames::metaDataTreeName().c_str())));
    if(nullptr == metaDataTree.get()) {
      throw Exception(errors::FileReadError) << "Could not find tree " << poolNames::metaDataTreeName()
                                             << " in the input file.\n";
    }

    // To keep things simple, we just read in every possible branch that exists.
    // We don't pay attention to which branches exist in which file format versions

    FileFormatVersion *fftPtr = &fileFormatVersion_;
    if(metaDataTree->FindBranch(poolNames::fileFormatVersionBranchName().c_str()) != nullptr) {
      TBranch *fft = metaDataTree->GetBranch(poolNames::fileFormatVersionBranchName().c_str());
      fft->SetAddress(&fftPtr);
      roottree::getEntry(fft, 0);
      metaDataTree->SetBranchAddress(poolNames::fileFormatVersionBranchName().c_str(), &fftPtr);
    }

    FileID *fidPtr = &fid_;
    if(metaDataTree->FindBranch(poolNames::fileIdentifierBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::fileIdentifierBranchName().c_str(), &fidPtr);
    }

    IndexIntoFile *iifPtr = &indexIntoFile_;
    if(metaDataTree->FindBranch(poolNames::indexIntoFileBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::indexIntoFileBranchName().c_str(), &iifPtr);
    }

    // Need to read to a temporary registry so we can do a translation of the BranchKeys.
    // This preserves backward compatibility against friendly class name algorithm changes.
    ProductRegistry inputProdDescReg;
    ProductRegistry *ppReg = &inputProdDescReg;
    metaDataTree->SetBranchAddress(poolNames::productDescriptionBranchName().c_str(), (&ppReg));

    typedef std::map<ParameterSetID, ParameterSetBlob> PsetMap;
    PsetMap psetMap;
    PsetMap *psetMapPtr = &psetMap;
    if(metaDataTree->FindBranch(poolNames::parameterSetMapBranchName().c_str()) != nullptr) {
      //backward compatibility
      assert(!fileFormatVersion().parameterSetsTree());
      metaDataTree->SetBranchAddress(poolNames::parameterSetMapBranchName().c_str(), &psetMapPtr);
    } else {
      assert(fileFormatVersion().parameterSetsTree());
      // We use a smart pointer so the tree will be deleted after use, and not kept for the life of the file.
      std::unique_ptr<TTree> psetTree(dynamic_cast<TTree *>(filePtr_->Get(poolNames::parameterSetsTreeName().c_str())));
      if(nullptr == psetTree.get()) {
        throw Exception(errors::FileReadError) << "Could not find tree " << poolNames::parameterSetsTreeName()
        << " in the input file.\n";
      }

      typedef std::pair<ParameterSetID, ParameterSetBlob> IdToBlobs;
      IdToBlobs idToBlob;
      IdToBlobs* pIdToBlob = &idToBlob;
      psetTree->SetBranchAddress(poolNames::idToParameterSetBlobsBranchName().c_str(), &pIdToBlob);

      std::unique_ptr<TTreeCache> psetTreeCache = roottree::trainCache(psetTree.get(), *filePtr_, roottree::defaultNonEventCacheSize, "*");
      psetTreeCache->SetEnablePrefetching(false);
      filePtr_->SetCacheRead(psetTreeCache.get());
      for(Long64_t i = 0; i != psetTree->GetEntries(); ++i) {
        psetTree->GetEntry(i);
        psetMap.insert(idToBlob);
      }
      filePtr_->SetCacheRead(0);
    }

    // backward compatibility
    ProcessHistoryRegistry::collection_type pHistMap;
    ProcessHistoryRegistry::collection_type *pHistMapPtr = &pHistMap;
    if(metaDataTree->FindBranch(poolNames::processHistoryMapBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::processHistoryMapBranchName().c_str(), &pHistMapPtr);
    }

    ProcessHistoryRegistry::vector_type pHistVector;
    ProcessHistoryRegistry::vector_type *pHistVectorPtr = &pHistVector;
    if(metaDataTree->FindBranch(poolNames::processHistoryBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::processHistoryBranchName().c_str(), &pHistVectorPtr);
    }

    ProcessConfigurationVector* procConfigVectorPtr = &processConfigurations_;
    if(metaDataTree->FindBranch(poolNames::processConfigurationBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::processConfigurationBranchName().c_str(), &procConfigVectorPtr);
    }

    std::unique_ptr<BranchIDLists> branchIDListsAPtr(new BranchIDLists);
    BranchIDLists* branchIDListsPtr = branchIDListsAPtr.get();
    if(metaDataTree->FindBranch(poolNames::branchIDListBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::branchIDListBranchName().c_str(), &branchIDListsPtr);
    }

    BranchChildren* branchChildrenBuffer = branchChildren_.get();
    if(metaDataTree->FindBranch(poolNames::productDependenciesBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::productDependenciesBranchName().c_str(), &branchChildrenBuffer);
    }

    // backward compatibility
    std::vector<EventProcessHistoryID> *eventHistoryIDsPtr = &eventProcessHistoryIDs_;
    if(metaDataTree->FindBranch(poolNames::eventHistoryBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::eventHistoryBranchName().c_str(), &eventHistoryIDsPtr);
    }

    if(metaDataTree->FindBranch(poolNames::moduleDescriptionMapBranchName().c_str()) != nullptr) {
      if(metaDataTree->GetBranch(poolNames::moduleDescriptionMapBranchName().c_str())->GetSplitLevel() != 0) {
        metaDataTree->SetBranchStatus((poolNames::moduleDescriptionMapBranchName() + ".*").c_str(), 0);
      } else {
        metaDataTree->SetBranchStatus(poolNames::moduleDescriptionMapBranchName().c_str(), 0);
      }
    }

    // Here we read the metadata tree
    roottree::getEntry(metaDataTree.get(), 0);

    checkReleaseVersion();

    eventProcessHistoryIter_ = eventProcessHistoryIDs_.begin();

    // Here we read the event history tree, if we have one.
    readEventHistoryTree();

    ParameterSetConverter::ParameterSetIdConverter psetIdConverter;
    if(!fileFormatVersion().triggerPathsTracked()) {
      ParameterSetConverter converter(psetMap, psetIdConverter, fileFormatVersion().parameterSetsByReference());
    } else {
      // Merge into the parameter set registry.
      pset::Registry& psetRegistry = *pset::Registry::instance();
      for(auto const& psetEntry : psetMap) {
        ParameterSet pset(psetEntry.second.pset());
        pset.setID(psetEntry.first);
        psetRegistry.insertMapped(pset);
      }
    }
    if(!fileFormatVersion().splitProductIDs()) {
      // Old provenance format input file.  Create a provenance adaptor.
      provenanceAdaptor_.reset(new ProvenanceAdaptor(
            inputProdDescReg, pHistMap, pHistVector, processConfigurations_, psetIdConverter, true));
      // Fill in the branchIDLists branch from the provenance adaptor
      branchIDLists_ = provenanceAdaptor_->branchIDLists();
    } else {
      if(!fileFormatVersion().triggerPathsTracked()) {
        // New provenance format, but change in ParameterSet Format. Create a provenance adaptor.
        provenanceAdaptor_.reset(new ProvenanceAdaptor(
            inputProdDescReg, pHistMap, pHistVector, processConfigurations_, psetIdConverter, false));
      }
      // New provenance format input file. The branchIDLists branch was read directly from the input file.
      if(metaDataTree->FindBranch(poolNames::branchIDListBranchName().c_str()) == nullptr) {
        throw Exception(errors::EventCorruption)
          << "Failed to find branchIDLists branch in metaData tree.\n";
      }
      branchIDLists_.reset(branchIDListsAPtr.release());
    }

    if(labelRawDataLikeMC) {
      std::string const rawData("FEDRawDataCollection");
      std::string const source("source");
      ProductRegistry::ProductList& pList = inputProdDescReg.productListUpdator();
      BranchKey finder(rawData, source, "", "");
      ProductRegistry::ProductList::iterator it = pList.lower_bound(finder);
      if(it != pList.end() && it->first.friendlyClassName_ == rawData && it->first.moduleLabel_ == source) {
        // We found raw data with a module label of source.
        // We need to change the module label and process name.
        // Create helper.
        it->second.init();
        daqProvenanceHelper_.reset(new DaqProvenanceHelper(it->second.unwrappedTypeID()));
        // Create the new branch description
        BranchDescription const& newBD = daqProvenanceHelper_->constBranchDescription_.me();
        // Save info from the old and new branch descriptions
        daqProvenanceHelper_->saveInfo(it->second, newBD);
        // Map the new branch name to the old branch name.
        it->second.init();
        newBranchToOldBranch_.insert(std::make_pair(newBD.branchName(), it->second.branchName()));
        // Remove the old branch description from the product Registry.
        pList.erase(it);
        // Check that there was only one.
        it = pList.lower_bound(finder);
        assert(!(it != pList.end() && it->first.friendlyClassName_ == rawData && it->first.moduleLabel_ == source));
        // Insert the new branch description into the product registry.
        inputProdDescReg.copyProduct(newBD);
        // Fix up other per file metadata.
        daqProvenanceHelper_->fixMetaData(processConfigurations_, pHistVector);
        daqProvenanceHelper_->fixMetaData(*branchIDLists_);
        daqProvenanceHelper_->fixMetaData(*branchChildren_);
      }
    }

    ProcessHistoryRegistry::instance()->insertCollection(pHistVector);
    ProcessConfigurationRegistry::instance()->insertCollection(processConfigurations_);

    eventTree_.trainCache(BranchTypeToAuxiliaryBranchName(InEvent).c_str());

    validateFile(inputType, usingGoToEvent);

    // Read the parentage tree.  Old format files are handled internally in readParentageTree().
    readParentageTree();

    // Merge into the hashed registries.
    if(eventSkipperByID_ && eventSkipperByID_->somethingToSkip()) {
      whyNotFastClonable_ += FileBlock::EventsOrLumisSelectedByID;
    }

    initializeDuplicateChecker(indexesIntoFiles, currentIndexIntoFile);
    indexIntoFileIter_ = indexIntoFileBegin_ = indexIntoFile_.begin(noEventSort ? IndexIntoFile::firstAppearanceOrder : IndexIntoFile::numericalOrder);
    indexIntoFileEnd_ = indexIntoFile_.end(noEventSort ? IndexIntoFile::firstAppearanceOrder : IndexIntoFile::numericalOrder);
    forcedRunOffset_ = forcedRunOffset(forcedRunNumber, indexIntoFileBegin_, indexIntoFileEnd_);
    eventProcessHistoryIter_ = eventProcessHistoryIDs_.begin();

    // Set product presence information in the product registry.
    ProductRegistry::ProductList const& pList = inputProdDescReg.productList();
    for(auto const& product : pList) {
      BranchDescription const& prod = product.second;
      prod.init();
      treePointers_[prod.branchType()]->setPresence(prod, newBranchToOldBranch(prod.branchName()));
    }

    fillProductRegistryTransients(processConfigurations_, inputProdDescReg);

    std::unique_ptr<ProductRegistry> newReg(new ProductRegistry);

    // Do the translation from the old registry to the new one
    {
      ProductRegistry::ProductList const& prodList = inputProdDescReg.productList();
      for(auto const& product : prodList) {
        BranchDescription const& prod = product.second;
        std::string newFriendlyName = friendlyname::friendlyName(prod.className());
        if(newFriendlyName == prod.friendlyClassName()) {
          newReg->copyProduct(prod);
        } else {
          if(fileFormatVersion().splitProductIDs()) {
            throw Exception(errors::UnimplementedFeature)
              << "Cannot change friendly class name algorithm without more development work\n"
              << "to update BranchIDLists.  Contact the framework group.\n";
          }
          BranchDescription newBD(prod);
          newBD.updateFriendlyClassName();
          newReg->copyProduct(newBD);
          newBranchToOldBranch_.insert(std::make_pair(newBD.branchName(), prod.branchName()));
        }
      }
      dropOnInput(*newReg, productSelectorRules, dropDescendants, inputType);
      // freeze the product registry
      newReg->setFrozen(inputType != InputType::Primary);
      productRegistry_.reset(newReg.release());
    }

    // Here, we make the class that will make the ProvenanceReader
    provenanceReaderMaker_.reset(makeProvenanceReaderMaker().release());

    // Set up information from the product registry.
    ProductRegistry::ProductList const& prodList = productRegistry()->productList();
    for(auto const& product : prodList) {
      BranchDescription const& prod = product.second;
      treePointers_[prod.branchType()]->addBranch(product.first, prod,
                                                  newBranchToOldBranch(prod.branchName()));
    }

    // Event Principal cache for secondary input source
    if(inputType == InputType::SecondarySource) {
      secondaryEventPrincipal_.reset(new EventPrincipal(productRegistry(), branchIDListHelper_, processConfiguration, nullptr));
    }

    // Determine if this file is fast clonable.
    setIfFastClonable(remainingEvents, remainingLumis);

    // Update the branch id info.
    if(inputType == InputType::Primary) {
      branchListIndexesUnchanged_ = branchIDListHelper_->updateFromInput(*branchIDLists_);
    }

    setRefCoreStreamer(true);  // backward compatibility

    // We are done with our initial reading of EventAuxiliary.
    indexIntoFile_.doneFileInitialization();

    // Tell the event tree to begin training at the next read.
    eventTree_.resetTraining();

    // Train the run and lumi trees.
    runTree_.trainCache("*");
    lumiTree_.trainCache("*");
  }

  RootFile::~RootFile() {
  }

  void
  RootFile::readEntryDescriptionTree() {
    // Called only for old format files.
    if(!fileFormatVersion().perEventProductIDs()) return;
    // We use a smart pointer so the tree will be deleted after use, and not kept for the life of the file.
    std::unique_ptr<TTree> entryDescriptionTree(dynamic_cast<TTree*>(filePtr_->Get(poolNames::entryDescriptionTreeName().c_str())));
    if(nullptr == entryDescriptionTree.get()) {
      throw Exception(errors::FileReadError) << "Could not find tree " << poolNames::entryDescriptionTreeName()
                                             << " in the input file.\n";
    }

    EntryDescriptionID idBuffer;
    EntryDescriptionID* pidBuffer = &idBuffer;
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionIDBranchName().c_str(), &pidBuffer);

    EntryDescriptionRegistry& oldregistry = *EntryDescriptionRegistry::instance();

    EventEntryDescription entryDescriptionBuffer;
    EventEntryDescription *pEntryDescriptionBuffer = &entryDescriptionBuffer;
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionBranchName().c_str(), &pEntryDescriptionBuffer);

    // Fill in the parentage registry.
    ParentageRegistry& registry = *ParentageRegistry::instance();

    for(Long64_t i = 0, numEntries = entryDescriptionTree->GetEntries(); i < numEntries; ++i) {
      roottree::getEntry(entryDescriptionTree.get(), i);
      if(idBuffer != entryDescriptionBuffer.id()) {
        throw Exception(errors::EventCorruption) << "Corruption of EntryDescription tree detected.\n";
      }
      oldregistry.insertMapped(entryDescriptionBuffer);
      Parentage parents;
      parents.parents() = entryDescriptionBuffer.parents();
      if(daqProvenanceHelper_) {
        ParentageID const oldID = parents.id();
        daqProvenanceHelper_->fixMetaData(parents.parents());
        ParentageID newID = parents.id();
        if(newID != oldID) {
          daqProvenanceHelper_->parentageIDMap_.insert(std::make_pair(oldID, newID));
        }
      }
      registry.insertMapped(parents);
    }
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionIDBranchName().c_str(), nullptr);
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionBranchName().c_str(), nullptr);
  }

  void
  RootFile::readParentageTree() {
    if(!fileFormatVersion().splitProductIDs()) {
      // Old format file.
      readEntryDescriptionTree();
      return;
    }
    // New format file
    // We use a smart pointer so the tree will be deleted after use, and not kept for the life of the file.
    std::unique_ptr<TTree> parentageTree(dynamic_cast<TTree*>(filePtr_->Get(poolNames::parentageTreeName().c_str())));
    if(nullptr == parentageTree.get()) {
      throw Exception(errors::FileReadError) << "Could not find tree " << poolNames::parentageTreeName()
                                             << " in the input file.\n";
    }

    Parentage parents;
    Parentage *pParentageBuffer = &parents;
    parentageTree->SetBranchAddress(poolNames::parentageBranchName().c_str(), &pParentageBuffer);

    ParentageRegistry& registry = *ParentageRegistry::instance();

    parentageIDLookup_.reserve(parentageTree->GetEntries());
    for(Long64_t i = 0, numEntries = parentageTree->GetEntries(); i < numEntries; ++i) {
      roottree::getEntry(parentageTree.get(), i);
      if(daqProvenanceHelper_) {
        ParentageID const oldID = parents.id();
        daqProvenanceHelper_->fixMetaData(parents.parents());
        ParentageID newID = parents.id();
        if(newID != oldID) {
          daqProvenanceHelper_->parentageIDMap_.insert(std::make_pair(oldID, newID));
        }
      }
      registry.insertMapped(parents);
      parentageIDLookup_.push_back(parents.id());
    }
    parentageTree->SetBranchAddress(poolNames::parentageBranchName().c_str(), nullptr);
  }

  void
  RootFile::setIfFastClonable(int remainingEvents, int remainingLumis) {
    if(fileFormatVersion().noMetaDataTrees() and !fileFormatVersion().storedProductProvenanceUsed()) {
      //we must avoid copying the old branch which stored the per product per event provenance
      whyNotFastClonable_ += FileBlock::FileTooOld;
      return;
    }
    if(!fileFormatVersion().splitProductIDs()) {
      whyNotFastClonable_ += FileBlock::FileTooOld;
      return;
    }
    if(processingMode_ != InputSource::RunsLumisAndEvents) {
      whyNotFastClonable_ += FileBlock::NotProcessingEvents;
      return;
    }
    // Find entry for first event in file
    IndexIntoFile::IndexIntoFileItr it = indexIntoFileBegin_;
    while(it != indexIntoFileEnd_ && it.getEntryType() != IndexIntoFile::kEvent) {
      ++it;
    }
    if(it == indexIntoFileEnd_) {
      whyNotFastClonable_ += FileBlock::NoEventsInFile;
      return;
    }

    // From here on, record all reasons we can't fast clone.
    IndexIntoFile::SortOrder sortOrder = (noEventSort_ ? IndexIntoFile::firstAppearanceOrder : IndexIntoFile::numericalOrder);
    if(!indexIntoFile_.iterationWillBeInEntryOrder(sortOrder)) {
      whyNotFastClonable_ += (noEventSort_ ? FileBlock::RunOrLumiNotContiguous : FileBlock::EventsToBeSorted);
    }
    if(skipAnyEvents_) {
      whyNotFastClonable_ += FileBlock::InitialEventsSkipped;
    }
    if(remainingEvents >= 0 && eventTree_.entries() > remainingEvents) {
      whyNotFastClonable_ += FileBlock::MaxEventsTooSmall;
    }
    if(remainingLumis >= 0 && lumiTree_.entries() > remainingLumis) {
      whyNotFastClonable_ += FileBlock::MaxLumisTooSmall;
    }
    // We no longer fast copy the EventAuxiliary branch, so there
    // is no longer any need to disable fast copying because the run
    // number is being modified.   Also, this check did not work anyway
    // because this function is called before forcedRunOffset_ is set.

    // if(forcedRunOffset_ != 0) {
    //   whyNotFastClonable_ += FileBlock::RunNumberModified;
    // }
    if(duplicateChecker_ &&
      !duplicateChecker_->checkDisabled() &&
      !duplicateChecker_->noDuplicatesInFile()) {
      whyNotFastClonable_ += FileBlock::DuplicateEventsRemoved;
    }
  }

  boost::shared_ptr<FileBlock>
  RootFile::createFileBlock() const {
    return boost::shared_ptr<FileBlock>(new FileBlock(fileFormatVersion(),
                                                     eventTree_.tree(),
                                                     eventTree_.metaTree(),
                                                     lumiTree_.tree(),
                                                     lumiTree_.metaTree(),
                                                     runTree_.tree(),
                                                     runTree_.metaTree(),
                                                     whyNotFastClonable(),
                                                     hasNewlyDroppedBranch(),
                                                     file_,
                                                     branchListIndexesUnchanged(),
                                                     modifiedIDs(),
                                                     branchChildren_,
                                                     branchIDLists_));
  }

  std::string const&
  RootFile::newBranchToOldBranch(std::string const& newBranch) const {
    std::map<std::string, std::string>::const_iterator it = newBranchToOldBranch_.find(newBranch);
    if(it != newBranchToOldBranch_.end()) {
      return it->second;
    }
    return newBranch;
  }

  IndexIntoFile::IndexIntoFileItr
  RootFile::indexIntoFileIter() const {
    return indexIntoFileIter_;
  }

  void
  RootFile::setPosition(IndexIntoFile::IndexIntoFileItr const& position) {
    indexIntoFileIter_.copyPosition(position);
  }

  bool
  RootFile::skipThisEntry() {
    if(indexIntoFileIter_ == indexIntoFileEnd_) {
        return false;
    }
    if(eventSkipperByID_ && eventSkipperByID_->somethingToSkip()) {

      // See first if the entire lumi or run is skipped, so we won't have to read the event Auxiliary in that case.
      if(eventSkipperByID_->skipIt(indexIntoFileIter_.run(), indexIntoFileIter_.lumi(), 0U)) {
        return true;
      }

      // The Lumi is not skipped.  If this is an event, see if the event is skipped.
      if(indexIntoFileIter_.getEntryType() == IndexIntoFile::kEvent) {
        fillEventAuxiliary();
        if(eventSkipperByID_->skipIt(indexIntoFileIter_.run(),
                                      indexIntoFileIter_.lumi(),
                                      eventAux_.id().event())) {
          return true;
        }
      }

      // Skip runs with no lumis if either lumisToSkip or lumisToProcess have been set to select lumis
      if(indexIntoFileIter_.getEntryType() == IndexIntoFile::kRun &&
          eventSkipperByID_->skippingLumis()) {
        IndexIntoFile::IndexIntoFileItr iterLumi = indexIntoFileIter_;

        // There are no lumis in this run, not even ones we will skip
        if(iterLumi.peekAheadAtLumi() == IndexIntoFile::invalidLumi) {
          return true;
        }
        // If we get here there are lumis in the run, check to see if we are skipping all of them
        do {
          if(!eventSkipperByID_->skipIt(iterLumi.run(), iterLumi.peekAheadAtLumi(), 0U)) {
            return false;
          }
        }
        while(iterLumi.skipLumiInRun());
        return true;
      }
    }
    return false;
  }

  IndexIntoFile::EntryType
  RootFile::getEntryTypeWithSkipping() {
    while(skipThisEntry()) {
      if(indexIntoFileIter_.getEntryType() == IndexIntoFile::kRun) {
        indexIntoFileIter_.advanceToNextRun();
      }
      else if(indexIntoFileIter_.getEntryType() == IndexIntoFile::kLumi) {
        indexIntoFileIter_.advanceToNextLumiOrRun();
      }
      else {
        ++indexIntoFileIter_;
      }
    }
    return indexIntoFileIter_.getEntryType();
  }

  bool
  RootFile::isDuplicateEvent() {
    assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kEvent);
    if(duplicateChecker_.get() == nullptr) {
      return false;
    }
    fillEventAuxiliary();
    return duplicateChecker_->isDuplicateAndCheckActive(indexIntoFileIter_.processHistoryIDIndex(),
        indexIntoFileIter_.run(), indexIntoFileIter_.lumi(), eventAux_.id().event(), file_);
  }

  IndexIntoFile::EntryType
  RootFile::getNextEntryTypeWanted() {
    IndexIntoFile::EntryType entryType = getEntryTypeWithSkipping();
    if(entryType == IndexIntoFile::kEnd) {
      return IndexIntoFile::kEnd;
    }
    if(entryType == IndexIntoFile::kRun) {
      return IndexIntoFile::kRun;
    } else if(processingMode_ == InputSource::Runs) {
      indexIntoFileIter_.advanceToNextRun();
      return getNextEntryTypeWanted();
    }
    if(entryType == IndexIntoFile::kLumi) {
      return IndexIntoFile::kLumi;
    } else if(processingMode_ == InputSource::RunsAndLumis) {
      indexIntoFileIter_.advanceToNextLumiOrRun();
      return getNextEntryTypeWanted();
    }
    if(isDuplicateEvent()) {
      ++indexIntoFileIter_;
      return getNextEntryTypeWanted();
    }
    return IndexIntoFile::kEvent;
  }

  bool
  RootFile::wasLastEventJustRead() const {
    IndexIntoFile::IndexIntoFileItr itr(indexIntoFileIter_);
    itr.advanceToEvent();
    return itr.getEntryType() == IndexIntoFile::kEnd;
  }

  bool
  RootFile::wasFirstEventJustRead() const {
    IndexIntoFile::IndexIntoFileItr itr(indexIntoFileIter_);
    int phIndex;
    RunNumber_t run;
    LuminosityBlockNumber_t lumi;
    IndexIntoFile::EntryNumber_t eventEntry;
    itr.skipEventBackward(phIndex,
                          run,
                          lumi,
                          eventEntry);
    itr.skipEventBackward(phIndex,
                          run,
                          lumi,
                          eventEntry);
    return eventEntry == IndexIntoFile::invalidEntry;
  }

  namespace {
    typedef IndexIntoFile::EntryNumber_t  EntryNumber_t;
    struct RunItem {
      RunItem(ProcessHistoryID const& phid, RunNumber_t const& run) :
        phid_(phid), run_(run) {}
      ProcessHistoryID phid_;
      RunNumber_t run_;
    };
    struct RunItemSortByRun {
      bool operator()(RunItem const& a, RunItem const& b) const {
        return a.run_ < b.run_;
      }
    };
    struct RunItemSortByRunPhid {
      bool operator()(RunItem const& a, RunItem const& b) const {
        return a.run_ < b.run_ || (!(b.run_ < a.run_) && a.phid_ < b.phid_);
      }
    };
    struct LumiItem {
      LumiItem(ProcessHistoryID const& phid, RunNumber_t const& run,
                 LuminosityBlockNumber_t const& lumi, EntryNumber_t const& entry) :
        phid_(phid), run_(run), lumi_(lumi), firstEventEntry_(entry),
        lastEventEntry_(entry == -1LL ? -1LL : entry + 1) {}
      ProcessHistoryID phid_;
      RunNumber_t run_;
      LuminosityBlockNumber_t lumi_;
      EntryNumber_t firstEventEntry_;
      EntryNumber_t lastEventEntry_;
    };
    struct LumiItemSortByRunLumi {
      bool operator()(LumiItem const& a, LumiItem const& b) const {
        return a.run_ < b.run_ || (!(b.run_ < a.run_) && a.lumi_ < b.lumi_);
      }
    };
    struct LumiItemSortByRunLumiPhid {
      bool operator()(LumiItem const& a, LumiItem const& b) const {
        if(a.run_ < b.run_) return true;
        if(b.run_ < a.run_) return false;
        if(a.lumi_ < b.lumi_) return true;
        if(b.lumi_ < a.lumi_) return false;
        return a.phid_ < b.phid_;
      }
    };
  }

  void
  RootFile::fillIndexIntoFile() {
    // This function is for backward compatibility.
    // If reading a current format file, indexIntoFile_ is read from the input
    // file and should always be there. Note that the algorithm below will work
    // sometimes but often fail with the new format introduced in release 3_8_0.
    // If it ever becomes necessary to rebuild IndexIntoFile from the new format,
    // probably a separate function should be written to deal with the task.
    // This is possible just not implemented yet.
    assert(!fileFormatVersion().hasIndexIntoFile());

    typedef std::list<LumiItem> LumiList;
    LumiList lumis; // (declare 1)

    typedef std::set<LuminosityBlockID> RunLumiSet;
    RunLumiSet runLumiSet; // (declare 2)

    typedef std::list<RunItem> RunList;
    RunList runs; // (declare 5)

    typedef std::set<RunNumber_t> RunSet;
    RunSet runSet; // (declare 4)

    typedef std::set<RunItem, RunItemSortByRunPhid> RunItemSet;
    RunItemSet runItemSet; // (declare 3)

    typedef std::map<RunNumber_t, ProcessHistoryID> PHIDMap;
    PHIDMap phidMap;

    RunNumber_t prevRun = 0;
    LuminosityBlockNumber_t prevLumi = 0;
    ProcessHistoryID prevPhid;
    bool iFirst = true;

    indexIntoFile_.unsortedEventNumbers().clear(); // should already be empty, just being careful
    indexIntoFile_.unsortedEventNumbers().reserve(eventTree_.entries());

    // First, loop through the event tree.
    while(eventTree_.next()) {
      bool newRun = false;
      bool newLumi = false;
      fillThisEventAuxiliary();
      fillHistory();

      // Save the event numbers as we loop through the event auxiliary to avoid
      // having to read through the event auxiliary again later. These event numbers
      // are not actually used in this function, but could be needed elsewhere.
      indexIntoFile_.unsortedEventNumbers().push_back(eventAux().event());

      ProcessHistoryID reducedPHID = ProcessHistoryRegistry::instance()->extra().reduceProcessHistoryID(eventAux().processHistoryID());

      if(iFirst || prevPhid != reducedPHID || prevRun != eventAux().run()) {
        iFirst = false;
        newRun = newLumi = true;
      } else if(prevLumi != eventAux().luminosityBlock()) {
        newLumi = true;
      }
      prevPhid = reducedPHID;
      prevRun = eventAux().run();
      prevLumi = eventAux().luminosityBlock();
      if(newLumi) {
        lumis.emplace_back(reducedPHID,
          eventAux().run(), eventAux().luminosityBlock(), eventTree_.entryNumber()); // (insert 1)
        runLumiSet.insert(LuminosityBlockID(eventAux().run(), eventAux().luminosityBlock())); // (insert 2)
      } else {
        LumiItem& currentLumi = lumis.back();
        assert(currentLumi.lastEventEntry_ == eventTree_.entryNumber());
        ++currentLumi.lastEventEntry_;
      }
      if(newRun) {
        // Insert run in list if it is not already there.
        RunItem item(reducedPHID, eventAux().run());
        if(runItemSet.insert(item).second) { // (check 3, insert 3)
          runs.push_back(std::move(item)); // (insert 5)
          runSet.insert(eventAux().run()); // (insert 4)
          phidMap.insert(std::make_pair(eventAux().run(), reducedPHID));
        }
      }
    }
    // now clean up.
    eventTree_.setEntryNumber(-1);
    eventAux_ = EventAuxiliary();
    lastEventEntryNumberRead_ = -1LL;

    // Loop over run entries and fill information.

    typedef std::map<RunNumber_t, EntryNumber_t> RunMap;
    RunMap runMap; // (declare 11)

    typedef std::vector<RunItem> RunVector;
    RunVector emptyRuns; // (declare 12)

    if(runTree_.isValid()) {
      while(runTree_.next()) {
        // Note: adjacent duplicates will be skipped without an explicit check.

        boost::shared_ptr<RunAuxiliary> runAux = fillRunAuxiliary();
        ProcessHistoryID reducedPHID = ProcessHistoryRegistry::instance()->extra().reduceProcessHistoryID(runAux->processHistoryID());

        if(runSet.insert(runAux->run()).second) { // (check 4, insert 4)
          // This run was not associated with any events.
          emptyRuns.emplace_back(reducedPHID, runAux->run()); // (insert 12)
        }
        runMap.insert(std::make_pair(runAux->run(), runTree_.entryNumber())); // (insert 11)
        phidMap.insert(std::make_pair(runAux->run(), reducedPHID));
      }
      // now clean up.
      runTree_.setEntryNumber(-1);
    }

    // Insert the ordered empty runs into the run list.
    RunItemSortByRun runItemSortByRun;
    stable_sort_all(emptyRuns, runItemSortByRun);

    RunList::iterator itRuns = runs.begin(), endRuns = runs.end();
    for(auto const& emptyRun : emptyRuns) {
      for(; itRuns != endRuns; ++itRuns) {
        if(runItemSortByRun(emptyRun, *itRuns)) {
          break;
        }
      }
      runs.insert(itRuns, emptyRun);
    }

    // Loop over luminosity block entries and fill information.

    typedef std::vector<LumiItem> LumiVector;
    LumiVector emptyLumis; // (declare 7)

    typedef std::map<LuminosityBlockID, EntryNumber_t> RunLumiMap;
    RunLumiMap runLumiMap; // (declare 6)

    if(lumiTree_.isValid()) {
      while(lumiTree_.next()) {
        // Note: adjacent duplicates will be skipped without an explicit check.
        boost::shared_ptr<LuminosityBlockAuxiliary> lumiAux = fillLumiAuxiliary();
        LuminosityBlockID lumiID = LuminosityBlockID(lumiAux->run(), lumiAux->luminosityBlock());
        if(runLumiSet.insert(lumiID).second) { // (check 2, insert 2)
          // This lumi was not associated with any events.
          // Use the process history ID from the corresponding run.  In cases of practical
          // importance, this should be the correct process history ID,  but it is possible
          // to construct files where this is not the correct process history ID ...
          PHIDMap::const_iterator iPhidMap = phidMap.find(lumiAux->run());
          assert(iPhidMap != phidMap.end());
          emptyLumis.emplace_back(iPhidMap->second, lumiAux->run(), lumiAux->luminosityBlock(), -1LL); // (insert 7)
        }
        runLumiMap.insert(std::make_pair(lumiID, lumiTree_.entryNumber()));
      }
      // now clean up.
      lumiTree_.setEntryNumber(-1);
    }

    // Insert the ordered empty lumis into the lumi list.
    LumiItemSortByRunLumi lumiItemSortByRunLumi;
    stable_sort_all(emptyLumis, lumiItemSortByRunLumi);

    LumiList::iterator itLumis = lumis.begin(), endLumis = lumis.end();
    for(auto const& emptyLumi : emptyLumis) {
      for(; itLumis != endLumis; ++itLumis) {
        if(lumiItemSortByRunLumi(emptyLumi, *itLumis)) {
          break;
        }
      }
      lumis.insert(itLumis, emptyLumi);
    }

    // Create a map of RunItems that gives the order of first appearance in the list.
    // Also fill in the vector of process history IDs
    typedef std::map<RunItem, int, RunItemSortByRunPhid> RunCountMap;
    RunCountMap runCountMap; // Declare (17)
    std::vector<ProcessHistoryID>& phids = indexIntoFile_.setProcessHistoryIDs();
    assert(phids.empty());
    std::vector<IndexIntoFile::RunOrLumiEntry>& entries = indexIntoFile_.setRunOrLumiEntries();
    assert(entries.empty());
    int rcount = 0;
    for(auto& run : runs) {
      RunCountMap::const_iterator countMapItem = runCountMap.find(run);
      if(countMapItem == runCountMap.end()) {
        countMapItem = runCountMap.insert(std::make_pair(run, rcount)).first; // Insert (17)
        assert(countMapItem != runCountMap.end());
        ++rcount;
      }
      std::vector<ProcessHistoryID>::const_iterator phidItem = find_in_all(phids, run.phid_);
      if(phidItem == phids.end()) {
        phids.push_back(run.phid_);
        phidItem = phids.end() - 1;
      }
      entries.emplace_back(
        countMapItem->second, // use (17)
        -1LL,
        runMap[run.run_], // use (11)
        phidItem - phids.begin(),
        run.run_,
        0U,
        -1LL,
        -1LL);
    }

    // Create a map of LumiItems that gives the order of first appearance in the list.
    typedef std::map<LumiItem, int, LumiItemSortByRunLumiPhid> LumiCountMap;
    LumiCountMap lumiCountMap; // Declare (19)
    int lcount = 0;
    for(auto& lumi : lumis) {
      RunCountMap::const_iterator runCountMapItem = runCountMap.find(RunItem(lumi.phid_, lumi.run_));
      assert(runCountMapItem != runCountMap.end());
      LumiCountMap::const_iterator countMapItem = lumiCountMap.find(lumi);
      if(countMapItem == lumiCountMap.end()) {
        countMapItem = lumiCountMap.insert(std::make_pair(lumi, lcount)).first; // Insert (17)
        assert(countMapItem != lumiCountMap.end());
        ++lcount;
      }
      std::vector<ProcessHistoryID>::const_iterator phidItem = find_in_all(phids, lumi.phid_);
      assert(phidItem != phids.end());
      entries.emplace_back(
        runCountMapItem->second,
        countMapItem->second,
        runLumiMap[LuminosityBlockID(lumi.run_, lumi.lumi_)],
        phidItem - phids.begin(),
        lumi.run_,
        lumi.lumi_,
        lumi.firstEventEntry_,
        lumi.lastEventEntry_);
    }
    stable_sort_all(entries);
  }

  void
  RootFile::validateFile(InputType::InputType inputType, bool usingGoToEvent) {
    if(!fid_.isValid()) {
      fid_ = FileID(createGlobalIdentifier());
    }
    if(!eventTree_.isValid()) {
      throw Exception(errors::EventCorruption) <<
         "'Events' tree is corrupted or not present\n" << "in the input file.\n";
    }

    if(fileFormatVersion().hasIndexIntoFile()) {
      if(runTree().entries() > 0) {
        assert(!indexIntoFile_.empty());
      }
      if(!fileFormatVersion().useReducedProcessHistoryID()) {
        if(daqProvenanceHelper_) {
          std::vector<ProcessHistoryID>& phidVec = indexIntoFile_.setProcessHistoryIDs();
          for(auto& phid : phidVec) {
            phid = daqProvenanceHelper_->mapProcessHistoryID(phid);
          }
        }
        indexIntoFile_.reduceProcessHistoryIDs();
      }
    }
    else {
      assert(indexIntoFile_.empty());
      fillIndexIntoFile();
    }

    indexIntoFile_.fixIndexes(orderedProcessHistoryIDs_);
    indexIntoFile_.setNumberOfEvents(eventTree_.entries());
    indexIntoFile_.setEventFinder(boost::shared_ptr<IndexIntoFile::EventFinder>(new RootFileEventFinder(eventTree_)));
    // We fill the event numbers explicitly if we need to find events in closed files,
    // such as for secondary files (or secondary sources) or if duplicate checking across files.
    bool needEventNumbers = false;
    bool needIndexesForDuplicateChecker = duplicateChecker_ && duplicateChecker_->checkingAllFiles() && !duplicateChecker_->checkDisabled();
    if(inputType != InputType::Primary || needIndexesForDuplicateChecker || usingGoToEvent) {
      needEventNumbers = true;
    }
    bool needEventEntries = false;
    if(inputType != InputType::Primary || !noEventSort_) {
      // We need event entries for sorting or for secondary files or sources.
      needEventEntries = true;
    }
    indexIntoFile_.fillEventNumbersOrEntries(needEventNumbers, needEventEntries);
  }

  void
  RootFile::reportOpened(std::string const& inputType) {
    // Report file opened.
    std::string const label = "source";
    std::string moduleName = "PoolSource";
    filePtr_->inputFileOpened(
              logicalFile_,
              inputType,
              moduleName,
              label,
              fid_.fid(),
              eventTree_.branchNames());
  }

  void
  RootFile::close() {
    // Just to play it safe, zero all pointers to objects in the InputFile to be closed.
    eventHistoryTree_ = nullptr;
    for(auto& treePointer : treePointers_) {
      treePointer->close();
      treePointer = nullptr;
    }
    filePtr_->Close();
    filePtr_.reset();
  }

  void
  RootFile::fillThisEventAuxiliary() {
    if(lastEventEntryNumberRead_ == eventTree_.entryNumber()) {
      // Already read.
      return;
    }
    if(fileFormatVersion().newAuxiliary()) {
      EventAuxiliary *pEvAux = &eventAux_;
      eventTree_.fillAux<EventAuxiliary>(pEvAux);
    } else {
      // for backward compatibility.
      EventAux eventAux;
      EventAux *pEvAux = &eventAux;
      eventTree_.fillAux<EventAux>(pEvAux);
      conversion(eventAux, eventAux_);
    }
    lastEventEntryNumberRead_ = eventTree_.entryNumber();
  }

  void
  RootFile::fillEventAuxiliary() {
    eventTree_.setEntryNumber(indexIntoFileIter_.entry());
    fillThisEventAuxiliary();
  }

  void
  RootFile::fillHistory() {
    // We could consider doing delayed reading, but because we have to
    // store this History object in a different tree than the event
    // data tree, this is too hard to do in this first version.

    if(fileFormatVersion().eventHistoryBranch()) {
      // Lumi block number was not in EventID for the relevant releases.
      EventID id(eventAux().id().run(), 0, eventAux().id().event());
      if(eventProcessHistoryIter_->eventID() != id) {
        EventProcessHistoryID target(id, ProcessHistoryID());
        eventProcessHistoryIter_ = lower_bound_all(eventProcessHistoryIDs_, target);
        assert(eventProcessHistoryIter_->eventID() == id);
      }
      eventAux_.setProcessHistoryID(eventProcessHistoryIter_->processHistoryID());
      ++eventProcessHistoryIter_;
    } else if(fileFormatVersion().eventHistoryTree()) {
      // for backward compatibility.
      History* pHistory = history_.get();
      TBranch* eventHistoryBranch = eventHistoryTree_->GetBranch(poolNames::eventHistoryBranchName().c_str());
      if(!eventHistoryBranch) {
        throw Exception(errors::EventCorruption)
          << "Failed to find history branch in event history tree.\n";
      }
      eventHistoryBranch->SetAddress(&pHistory);
      roottree::getEntry(eventHistoryTree_, eventTree_.entryNumber());
      eventAux_.setProcessHistoryID(history_->processHistoryID());
      eventSelectionIDs_.reset(&history_->eventSelectionIDs(), do_nothing_deleter());
      branchListIndexes_.reset(&history_->branchListIndexes(), do_nothing_deleter());
    } else if(fileFormatVersion().noMetaDataTrees()) {
      // Current format
      EventSelectionIDVector* pESV = eventSelectionIDs_.get();
      TBranch* eventSelectionIDBranch = eventTree_.tree()->GetBranch(poolNames::eventSelectionsBranchName().c_str());
      assert(eventSelectionIDBranch != nullptr);
      eventTree_.fillBranchEntry(eventSelectionIDBranch, pESV);
      BranchListIndexes* pBLI = branchListIndexes_.get();
      TBranch* branchListIndexesBranch = eventTree_.tree()->GetBranch(poolNames::branchListIndexesBranchName().c_str());
      assert(branchListIndexesBranch != nullptr);
      eventTree_.fillBranchEntry(branchListIndexesBranch, pBLI);
    }
    if(provenanceAdaptor_) {
      eventAux_.setProcessHistoryID(provenanceAdaptor_->convertID(eventAux().processHistoryID()));
      for(auto& esID : *eventSelectionIDs_) {
        esID = provenanceAdaptor_->convertID(esID);
      }
    }
    if(daqProvenanceHelper_) {
      eventAux_.setProcessHistoryID(daqProvenanceHelper_->mapProcessHistoryID(eventAux_.processHistoryID()));
    }
    if(!fileFormatVersion().splitProductIDs()) {
      // old format.  branchListIndexes_ must be filled in from the ProvenanceAdaptor.
      provenanceAdaptor_->branchListIndexes(*branchListIndexes_);
    }
    branchIDListHelper_->fixBranchListIndexes(*branchListIndexes_);
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  RootFile::fillLumiAuxiliary() {
    boost::shared_ptr<LuminosityBlockAuxiliary> lumiAuxiliary(new LuminosityBlockAuxiliary);
    if(fileFormatVersion().newAuxiliary()) {
      LuminosityBlockAuxiliary *pLumiAux = lumiAuxiliary.get();
      lumiTree_.fillAux<LuminosityBlockAuxiliary>(pLumiAux);
    } else {
      LuminosityBlockAux lumiAux;
      LuminosityBlockAux *pLumiAux = &lumiAux;
      lumiTree_.fillAux<LuminosityBlockAux>(pLumiAux);
      conversion(lumiAux, *lumiAuxiliary);
    }
    if(provenanceAdaptor_) {
      lumiAuxiliary->setProcessHistoryID(provenanceAdaptor_->convertID(lumiAuxiliary->processHistoryID()));
    }
    if(daqProvenanceHelper_) {
      lumiAuxiliary->setProcessHistoryID(daqProvenanceHelper_->mapProcessHistoryID(lumiAuxiliary->processHistoryID()));
    }
    if(lumiAuxiliary->luminosityBlock() == 0 && !fileFormatVersion().runsAndLumis()) {
      lumiAuxiliary->id() = LuminosityBlockID(RunNumber_t(1), LuminosityBlockNumber_t(1));
    }
    return lumiAuxiliary;
  }

  boost::shared_ptr<RunAuxiliary>
  RootFile::fillRunAuxiliary() {
    boost::shared_ptr<RunAuxiliary> runAuxiliary(new RunAuxiliary);
    if(fileFormatVersion().newAuxiliary()) {
      RunAuxiliary *pRunAux = runAuxiliary.get();
      runTree_.fillAux<RunAuxiliary>(pRunAux);
    } else {
      RunAux runAux;
      RunAux *pRunAux = &runAux;
      runTree_.fillAux<RunAux>(pRunAux);
      conversion(runAux, *runAuxiliary);
    }
    if(provenanceAdaptor_) {
      runAuxiliary->setProcessHistoryID(provenanceAdaptor_->convertID(runAuxiliary->processHistoryID()));
    }
    if(daqProvenanceHelper_) {
      runAuxiliary->setProcessHistoryID(daqProvenanceHelper_->mapProcessHistoryID(runAuxiliary->processHistoryID()));
    }
    return runAuxiliary;
  }

  bool
  RootFile::skipEvents(int& offset) {
    while(offset > 0 && indexIntoFileIter_ != indexIntoFileEnd_) {

      int phIndexOfSkippedEvent = IndexIntoFile::invalidIndex;
      RunNumber_t runOfSkippedEvent = IndexIntoFile::invalidRun;
      LuminosityBlockNumber_t lumiOfSkippedEvent = IndexIntoFile::invalidLumi;
      IndexIntoFile::EntryNumber_t skippedEventEntry = IndexIntoFile::invalidEntry;

      indexIntoFileIter_.skipEventForward(phIndexOfSkippedEvent,
                                          runOfSkippedEvent,
                                          lumiOfSkippedEvent,
                                          skippedEventEntry);

      // At the end of the file and there were no more events to skip
      if(skippedEventEntry == IndexIntoFile::invalidEntry) break;

      if(eventSkipperByID_ && eventSkipperByID_->somethingToSkip()) {
        eventTree_.setEntryNumber(skippedEventEntry);
        fillThisEventAuxiliary();
        if(eventSkipperByID_->skipIt(runOfSkippedEvent, lumiOfSkippedEvent, eventAux_.id().event())) {
            continue;
        }
      }
      if(duplicateChecker_ &&
         !duplicateChecker_->checkDisabled() &&
         !duplicateChecker_->noDuplicatesInFile()) {

        eventTree_.setEntryNumber(skippedEventEntry);
        fillThisEventAuxiliary();
        if(duplicateChecker_->isDuplicateAndCheckActive(phIndexOfSkippedEvent,
                                                         runOfSkippedEvent,
                                                         lumiOfSkippedEvent,
                                                         eventAux_.id().event(),
                                                         file_)) {
          continue;
        }
      }
      --offset;
    }

    while(offset < 0) {

      if(duplicateChecker_) {
        duplicateChecker_->disable();
      }

      int phIndexOfEvent = IndexIntoFile::invalidIndex;
      RunNumber_t runOfEvent =  IndexIntoFile::invalidRun;
      LuminosityBlockNumber_t lumiOfEvent = IndexIntoFile::invalidLumi;
      EntryNumber_t eventEntry = IndexIntoFile::invalidEntry;

      indexIntoFileIter_.skipEventBackward(phIndexOfEvent,
                                           runOfEvent,
                                           lumiOfEvent,
                                           eventEntry);

      if(eventEntry == IndexIntoFile::invalidEntry) break;

      if(eventSkipperByID_ && eventSkipperByID_->somethingToSkip()) {
        eventTree_.setEntryNumber(eventEntry);
        fillEventAuxiliary();
        if(eventSkipperByID_->skipIt(runOfEvent, lumiOfEvent, eventAux_.id().event())) {
          continue;
        }
      }
      ++offset;
    }
    return(indexIntoFileIter_ == indexIntoFileEnd_);
  }

  bool
  RootFile::goToEvent(EventID const& eventID) {

    indexIntoFile_.fillEventNumbers();

    if(duplicateChecker_) {
      duplicateChecker_->disable();
    }

    IndexIntoFile::SortOrder sortOrder = IndexIntoFile::numericalOrder;
    if(noEventSort_) sortOrder = IndexIntoFile::firstAppearanceOrder;

    IndexIntoFile::IndexIntoFileItr iter =
      indexIntoFile_.findPosition(sortOrder, eventID.run(), eventID.luminosityBlock(), eventID.event());

    if(iter == indexIntoFile_.end(sortOrder)) {
      return false;
    }
    indexIntoFileIter_ = iter;
    return true;
  }

  // readEvent() is responsible for creating, and setting up, the
  // EventPrincipal.
  //
  //   1. create an EventPrincipal with a unique EventID
  //   2. For each entry in the provenance, put in one ProductHolder,
  //      holding the Provenance for the corresponding EDProduct.
  //   3. set up the caches in the EventPrincipal to know about this
  //      ProductHolder.
  //
  // We do *not* create the EDProduct instance (the equivalent of reading
  // the branch containing this EDProduct. That will be done by the Delayed Reader,
  //  when it is asked to do so.
  //
  EventPrincipal*
  RootFile::readEvent(EventPrincipal& cache) {
    assert(indexIntoFileIter_ != indexIntoFileEnd_);
    assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kEvent);
    // Set the entry in the tree, and read the event at that entry.
    eventTree_.setEntryNumber(indexIntoFileIter_.entry());
    EventPrincipal* ep = readCurrentEvent(cache);

    assert(ep != nullptr);
    assert(eventAux().run() == indexIntoFileIter_.run() + forcedRunOffset_);
    assert(eventAux().luminosityBlock() == indexIntoFileIter_.lumi());

    // If this next assert shows up in performance profiling or significantly affects memory, then these three lines should be deleted.
    // The IndexIntoFile should guarantee that it never fails.
    ProcessHistoryID idToCheck = (daqProvenanceHelper_ && fileFormatVersion().useReducedProcessHistoryID() ? *daqProvenanceHelper_->oldProcessHistoryID_ : eventAux().processHistoryID());
    ProcessHistoryID const& reducedPHID = ProcessHistoryRegistry::instance()->extra().reduceProcessHistoryID(idToCheck);
    assert(reducedPHID == indexIntoFile_.processHistoryID(indexIntoFileIter_.processHistoryIDIndex()));

    ++indexIntoFileIter_;
    return ep;
  }

  // Reads event at the current entry in the event tree
  EventPrincipal*
  RootFile::readCurrentEvent(EventPrincipal& cache) {
    if(!eventTree_.current()) {
      return nullptr;
    }
    fillThisEventAuxiliary();
    if(!fileFormatVersion().lumiInEventID()) {
        //ugly, but will disappear when the backward compatibility is done with schema evolution.
        const_cast<EventID&>(eventAux_.id()).setLuminosityBlockNumber(eventAux_.oldLuminosityBlock());
        eventAux_.resetObsoleteInfo();
    }
    fillHistory();
    overrideRunNumber(eventAux_.id(), eventAux().isRealData());

    // We're not done ... so prepare the EventPrincipal
    cache.fillEventPrincipal(eventAux(),
                             eventSelectionIDs_,
                             branchListIndexes_,
                             makeBranchMapper(),
                             eventTree_.rootDelayedReader());

    // report event read from file
    filePtr_->eventReadFromFile(eventID().run(), eventID().event());
    return &cache;
  }

  EventPrincipal*
  RootFile::clearAndReadCurrentEvent(EventPrincipal& cache) {
    cache.clearEventPrincipal();
    return readCurrentEvent(cache);
  }

  void
  RootFile::setAtEventEntry(IndexIntoFile::EntryNumber_t entry) {
    eventTree_.setEntryNumber(entry);
  }

  boost::shared_ptr<RunAuxiliary>
  RootFile::readRunAuxiliary_() {
    assert(indexIntoFileIter_ != indexIntoFileEnd_);
    assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kRun);

    // Begin code for backward compatibility before the existence of run trees.
    if(!runTree_.isValid()) {

      // prior to the support of run trees.
      // RunAuxiliary did not contain a valid timestamp.  Take it from the next event.
      IndexIntoFile::EntryNumber_t eventEntry = indexIntoFileIter_.firstEventEntryThisRun();
      assert(eventEntry != IndexIntoFile::invalidEntry);
      RootTree::EntryNumber savedEntry = eventTree_.entryNumber();
      eventTree_.setEntryNumber(eventEntry);
      assert(eventTree_.current());
      fillThisEventAuxiliary();
      eventTree_.setEntryNumber(savedEntry);

      RunID run = RunID(indexIntoFileIter_.run());
      overrideRunNumber(run);
      return boost::shared_ptr<RunAuxiliary>(new RunAuxiliary(run.run(), eventAux().time(), Timestamp::invalidTimestamp()));
    }
    // End code for backward compatibility before the existence of run trees.
    runTree_.setEntryNumber(indexIntoFileIter_.entry());
    boost::shared_ptr<RunAuxiliary> runAuxiliary = fillRunAuxiliary();
    assert(runAuxiliary->run() == indexIntoFileIter_.run());
    overrideRunNumber(runAuxiliary->id());
    filePtr_->reportInputRunNumber(runAuxiliary->run());
    // If RunAuxiliary did not contain a valid begin timestamp, invalidate any end timestamp.
    if(runAuxiliary->beginTime() == Timestamp::invalidTimestamp()) {
      runAuxiliary->setEndTime(Timestamp::invalidTimestamp());
    }

    // If RunAuxiliary did not contain a valid timestamp, or if this an old format file from
    // when the Run's ProcessHistory included only processes where products were added to the Run itself,
    // we attempt to read the first event in the run to get appropriate info.
    if(runAuxiliary->beginTime() == Timestamp::invalidTimestamp() ||
       !fileFormatVersion().processHistorySameWithinRun()) {

      IndexIntoFile::EntryNumber_t eventEntry = indexIntoFileIter_.firstEventEntryThisRun();
      // If we have a valid event, use its information.
      if(eventEntry != IndexIntoFile::invalidEntry) {
        RootTree::EntryNumber savedEntry = eventTree_.entryNumber();
        eventTree_.setEntryNumber(eventEntry);
        assert(eventTree_.current());
        fillThisEventAuxiliary();

        // RunAuxiliary did not contain a valid timestamp.  Take it from the next event in this run if there is one.
        if(runAuxiliary->beginTime() == Timestamp::invalidTimestamp()) {
          runAuxiliary->setBeginTime(eventAux().time());
        }

        // For backwards compatibility when the Run's ProcessHistory included only processes where products were added to the
        // Run, and then the Run and Event auxiliaries could be different.  Use the event ProcessHistoryID if there is one. It should
        // almost always be correct by the current definition (processes included if any products are added. This makes the run, lumi,
        // and event ProcessHistory's always be the same if no file merging occurs).
        if(!fileFormatVersion().processHistorySameWithinRun()) {
          fillHistory();
          runAuxiliary->setProcessHistoryID(eventAux().processHistoryID());
          savedRunAuxiliary_ = runAuxiliary;
        }
        eventTree_.setEntryNumber(savedEntry);
      } else {
        // No valid event, just use what is there, because it is the best we can do.
        savedRunAuxiliary_ = runAuxiliary;
      }
    }
    return runAuxiliary;
  }

  boost::shared_ptr<RunPrincipal>
  RootFile::readRun_(boost::shared_ptr<RunPrincipal> runPrincipal) {
    assert(indexIntoFileIter_ != indexIntoFileEnd_);
    assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kRun);
    // Begin code for backward compatibility before the existence of run trees.
    if(!runTree_.isValid()) {
      ++indexIntoFileIter_;
      return runPrincipal;
    }
    // End code for backward compatibility before the existence of run trees.
    runPrincipal->fillRunPrincipal(runTree_.rootDelayedReader());
    // Read in all the products now.
    runPrincipal->readImmediate();
    ++indexIntoFileIter_;
    return runPrincipal;
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  RootFile::readLuminosityBlockAuxiliary_() {
    assert(indexIntoFileIter_ != indexIntoFileEnd_);
    assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kLumi);
    // Begin code for backward compatibility before the existence of lumi trees.
    if(!lumiTree_.isValid()) {
      IndexIntoFile::EntryNumber_t eventEntry = indexIntoFileIter_.firstEventEntryThisLumi();
      assert(eventEntry != IndexIntoFile::invalidEntry);
      RootTree::EntryNumber savedEntry = eventTree_.entryNumber();
      eventTree_.setEntryNumber(eventEntry);
      assert(eventTree_.current());
      fillThisEventAuxiliary();
      eventTree_.setEntryNumber(savedEntry);

      LuminosityBlockID lumi = LuminosityBlockID(indexIntoFileIter_.run(), indexIntoFileIter_.lumi());
      overrideRunNumber(lumi);
      return boost::shared_ptr<LuminosityBlockAuxiliary>(new LuminosityBlockAuxiliary(lumi.run(), lumi.luminosityBlock(), eventAux().time(), Timestamp::invalidTimestamp()));
    }
    // End code for backward compatibility before the existence of lumi trees.
    lumiTree_.setEntryNumber(indexIntoFileIter_.entry());
    boost::shared_ptr<LuminosityBlockAuxiliary> lumiAuxiliary = fillLumiAuxiliary();
    assert(lumiAuxiliary->run() == indexIntoFileIter_.run());
    assert(lumiAuxiliary->luminosityBlock() == indexIntoFileIter_.lumi());
    overrideRunNumber(lumiAuxiliary->id());
    filePtr_->reportInputLumiSection(lumiAuxiliary->run(), lumiAuxiliary->luminosityBlock());
    if(lumiAuxiliary->beginTime() == Timestamp::invalidTimestamp()) {
      IndexIntoFile::EntryNumber_t eventEntry = indexIntoFileIter_.firstEventEntryThisLumi();
      if(eventEntry != IndexIntoFile::invalidEntry) {
        RootTree::EntryNumber savedEntry = eventTree_.entryNumber();
        eventTree_.setEntryNumber(eventEntry);
        assert(eventTree_.current());
        fillThisEventAuxiliary();
        eventTree_.setEntryNumber(savedEntry);

        lumiAuxiliary->setBeginTime(eventAux().time());
      }
      lumiAuxiliary->setEndTime(Timestamp::invalidTimestamp());
    }
    if(!fileFormatVersion().processHistorySameWithinRun() && savedRunAuxiliary_) {
      lumiAuxiliary->setProcessHistoryID(savedRunAuxiliary_->processHistoryID());
    }
    return lumiAuxiliary;
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  RootFile::readLumi(boost::shared_ptr<LuminosityBlockPrincipal> lumiPrincipal) {
    assert(indexIntoFileIter_ != indexIntoFileEnd_);
    assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kLumi);
    // Begin code for backward compatibility before the existence of lumi trees.
    if(!lumiTree_.isValid()) {
      ++indexIntoFileIter_;
      return lumiPrincipal;
    }
    // End code for backward compatibility before the existence of lumi trees.
    lumiTree_.setEntryNumber(indexIntoFileIter_.entry());
    lumiPrincipal->fillLuminosityBlockPrincipal(lumiTree_.rootDelayedReader());
    // Read in all the products now.
    lumiPrincipal->readImmediate();
    ++indexIntoFileIter_;
    return lumiPrincipal;
  }

  bool
  RootFile::setEntryAtEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) {
    indexIntoFileIter_ = indexIntoFile_.findEventPosition(run, lumi, event);
    if(indexIntoFileIter_ == indexIntoFileEnd_) return false;
    eventTree_.setEntryNumber(indexIntoFileIter_.entry());
    return true;
  }

  bool
  RootFile::setEntryAtLumi(RunNumber_t run, LuminosityBlockNumber_t lumi) {
    indexIntoFileIter_ = indexIntoFile_.findLumiPosition(run, lumi);
    if(indexIntoFileIter_ == indexIntoFileEnd_) return false;
    lumiTree_.setEntryNumber(indexIntoFileIter_.entry());
    return true;
  }

  bool
  RootFile::setEntryAtRun(RunNumber_t run) {
    indexIntoFileIter_ = indexIntoFile_.findRunPosition(run);
    if(indexIntoFileIter_ == indexIntoFileEnd_) return false;
    runTree_.setEntryNumber(indexIntoFileIter_.entry());
    return true;
  }

  bool
  RootFile::setEntryAtNextEventInLumi(RunNumber_t run, LuminosityBlockNumber_t lumi) {
    if(indexIntoFileIter_.getEntryType() == IndexIntoFile::kEvent) {
      ++indexIntoFileIter_;
    }
    indexIntoFileIter_.advanceToEvent();
    if(indexIntoFileIter_.getEntryType() != IndexIntoFile::kEvent) return false;
    if(run != indexIntoFileIter_.run()) return false;
    if(lumi != indexIntoFileIter_.lumi()) return false;
    eventTree_.setEntryNumber(indexIntoFileIter_.entry());
    return true;
  }

  void
  RootFile::overrideRunNumber(RunID& id) {
    if(forcedRunOffset_ != 0) {
      id = RunID(id.run() + forcedRunOffset_);
    }
    if(id < RunID::firstValidRun()) id = RunID::firstValidRun();
  }

  void
  RootFile::overrideRunNumber(LuminosityBlockID& id) {
    if(forcedRunOffset_ != 0) {
      id = LuminosityBlockID(id.run() + forcedRunOffset_, id.luminosityBlock());
    }
    if(RunID(id.run()) < RunID::firstValidRun()) id = LuminosityBlockID(RunID::firstValidRun().run(), id.luminosityBlock());
  }

  void
  RootFile::overrideRunNumber(EventID& id, bool isRealData) {
    if(forcedRunOffset_ != 0) {
      if(isRealData) {
        throw Exception(errors::Configuration, "RootFile::RootFile()")
          << "The 'setRunNumber' parameter of PoolSource cannot be used with real data.\n";
      }
      id = EventID(id.run() + forcedRunOffset_, id.luminosityBlock(), id.event());
    }
    if(RunID(id.run()) < RunID::firstValidRun()) {
      id = EventID(RunID::firstValidRun().run(), LuminosityBlockID::firstValidLuminosityBlock().luminosityBlock(), id.event());
    }
  }


  void
  RootFile::readEventHistoryTree() {
    // Read in the event history tree, if we have one...
    if(fileFormatVersion().eventHistoryTree()) {
      history_.reset(new History);
      eventHistoryTree_ = dynamic_cast<TTree*>(filePtr_->Get(poolNames::eventHistoryTreeName().c_str()));
      if(!eventHistoryTree_) {
        throw Exception(errors::EventCorruption)
          << "Failed to find the event history tree.\n";
      }
    }
  }

  void
  RootFile::checkReleaseVersion() {
    std::string releaseVersion = getReleaseVersion();
    releaseversion::DecomposedReleaseVersion currentRelease(releaseVersion);
    for(auto const& pc : processConfigurations_) {
      if(releaseversion::isEarlierRelease(currentRelease, pc.releaseVersion())) {
        throw Exception(errors::FormatIncompatibility)
          << "The release you are using, " << getReleaseVersion() << " , predates\n"
          << "a release (" << pc.releaseVersion() << ") used in writing the input file, " << file() <<".\n"
          << "Forward compatibility cannot be supported.\n";
      }
    }
  }

  void
  RootFile::initializeDuplicateChecker(
    std::vector<boost::shared_ptr<IndexIntoFile> > const& indexesIntoFiles,
    std::vector<boost::shared_ptr<IndexIntoFile> >::size_type currentIndexIntoFile) {
    if(duplicateChecker_) {
      if(eventTree_.next()) {
        fillThisEventAuxiliary();
        duplicateChecker_->inputFileOpened(eventAux().isRealData(),
                                           indexIntoFile_,
                                           indexesIntoFiles,
                                           currentIndexIntoFile);
      }
      eventTree_.setEntryNumber(-1);
    }
  }

  void
  RootFile::dropOnInput (ProductRegistry& reg, ProductSelectorRules const& rules, bool dropDescendants, InputType::InputType inputType) {
    // This is the selector for drop on input.
    ProductSelector productSelector;
    productSelector.initialize(rules, reg.allBranchDescriptions());

    ProductRegistry::ProductList& prodList = reg.productListUpdator();
    // Do drop on input. On the first pass, just fill in a set of branches to be dropped.
    std::set<BranchID> branchesToDrop;
    for(auto const& product : prodList) {
      BranchDescription const& prod = product.second;
      if(!productSelector.selected(prod)) {
        if(dropDescendants) {
          branchChildren_->appendToDescendants(prod.branchID(), branchesToDrop);
        } else {
          branchesToDrop.insert(prod.branchID());
        }
      }
    }

    // On this pass, actually drop the branches.
    std::set<BranchID>::const_iterator branchesToDropEnd = branchesToDrop.end();
    for(ProductRegistry::ProductList::iterator it = prodList.begin(), itEnd = prodList.end(); it != itEnd;) {
      BranchDescription const& prod = it->second;
      bool drop = branchesToDrop.find(prod.branchID()) != branchesToDropEnd;
      if(drop) {
        if(productSelector.selected(prod)) {
          LogWarning("RootFile")
            << "Branch '" << prod.branchName() << "' is being dropped from the input\n"
            << "of file '" << file_ << "' because it is dependent on a branch\n"
            << "that was explicitly dropped.\n";
        }
        treePointers_[prod.branchType()]->dropBranch(newBranchToOldBranch(prod.branchName()));
        hasNewlyDroppedBranch_[prod.branchType()] = true;
        ProductRegistry::ProductList::iterator icopy = it;
        ++it;
        prodList.erase(icopy);
      } else {
        ++it;
      }
    }

    // Drop on input mergeable run and lumi products, this needs to be invoked for secondary file input
    if(inputType == InputType::SecondaryFile) {
      TString tString;
      for(ProductRegistry::ProductList::iterator it = prodList.begin(), itEnd = prodList.end(); it != itEnd;) {
        BranchDescription const& prod = it->second;
        if(prod.branchType() != InEvent) {
          TClass *cp = gROOT->GetClass(prod.wrappedName().c_str());
          WrapperOwningHolder edp(cp->New(), prod.getInterface());
          if(edp.isMergeable()) {
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

  std::unique_ptr<MakeProvenanceReader>
  RootFile::makeProvenanceReaderMaker() const {
    if(fileFormatVersion_.storedProductProvenanceUsed()) {
      return std::unique_ptr<MakeProvenanceReader>(new MakeReducedProvenanceReader(parentageIDLookup_));
    } else if(fileFormatVersion_.splitProductIDs()) {
      return std::unique_ptr<MakeProvenanceReader>(new MakeFullProvenanceReader);
    } else if(fileFormatVersion_.perEventProductIDs()) {
      return std::unique_ptr<MakeProvenanceReader>(new MakeOldProvenanceReader);
    } else {
      return std::unique_ptr<MakeProvenanceReader>(new MakeDummyProvenanceReader);
    }
  }

  boost::shared_ptr<BranchMapper>
  RootFile::makeBranchMapper() {
    if(!eventBranchMapper_) {
      eventBranchMapper_.reset(new BranchMapper(provenanceReaderMaker_->makeReader(eventTree_, daqProvenanceHelper_.get())));
    }
    eventBranchMapper_->reset();
    return eventBranchMapper_;
  }

  class ReducedProvenanceReader : public ProvenanceReaderBase {
  public:
    ReducedProvenanceReader(RootTree* iRootTree, std::vector<ParentageID> const& iParentageIDLookup, DaqProvenanceHelper const* daqProvenanceHelper);
  private:
    virtual void readProvenance(BranchMapper const& mapper) const;
    RootTree* rootTree_;
    TBranch* provBranch_;
    StoredProductProvenanceVector provVector_;
    StoredProductProvenanceVector* pProvVector_;
    std::vector<ParentageID> const& parentageIDLookup_;
    DaqProvenanceHelper const* daqProvenanceHelper_;
  };

  ReducedProvenanceReader::ReducedProvenanceReader(
                                              RootTree* iRootTree,
                                              std::vector<ParentageID> const& iParentageIDLookup,
                                              DaqProvenanceHelper const* daqProvenanceHelper) :
      ProvenanceReaderBase(),
      rootTree_(iRootTree),
      pProvVector_(&provVector_),
      parentageIDLookup_(iParentageIDLookup),
      daqProvenanceHelper_(daqProvenanceHelper) {
    provBranch_ = rootTree_->tree()->GetBranch(BranchTypeToProductProvenanceBranchName(rootTree_->branchType()).c_str());
  }

  void
  ReducedProvenanceReader::readProvenance(BranchMapper const& mapper) const {
    ReducedProvenanceReader* me = const_cast<ReducedProvenanceReader*>(this);
    me->rootTree_->fillBranchEntry(me->provBranch_, me->pProvVector_);
    setRefCoreStreamer(true);
    if(daqProvenanceHelper_) {
      for(auto const& prov : provVector_) {
        BranchID bid(prov.branchID_);
        mapper.insertIntoSet(ProductProvenance(daqProvenanceHelper_->mapBranchID(BranchID(prov.branchID_)),
                                               daqProvenanceHelper_->mapParentageID(parentageIDLookup_[prov.parentageIDIndex_])));
      }
    } else {
      for(auto const& prov : provVector_) {
        if(prov.parentageIDIndex_ >= parentageIDLookup_.size()) {
          throw edm::Exception(errors::LogicError)
            << "ReducedProvenanceReader::ReadProvenance\n"
            << "The parentage ID index value " << prov.parentageIDIndex_ << " is out of bounds.  The maximum value is " << parentageIDLookup_.size()-1 << ".\n"
            << "This should never happen.\n"
            << "Please report this to the framework hypernews forum 'hn-cms-edmFramework@cern.ch'.\n";
        }
        mapper.insertIntoSet(ProductProvenance(BranchID(prov.branchID_), parentageIDLookup_[prov.parentageIDIndex_]));
      }
    }
  }

  class FullProvenanceReader : public ProvenanceReaderBase {
  public:
    explicit FullProvenanceReader(RootTree* rootTree, DaqProvenanceHelper const* daqProvenanceHelper);
    virtual ~FullProvenanceReader() {}
  private:
    virtual void readProvenance(BranchMapper const& mapper) const;
    RootTree* rootTree_;
    ProductProvenanceVector infoVector_;
    mutable ProductProvenanceVector* pInfoVector_;
    DaqProvenanceHelper const* daqProvenanceHelper_;
  };

  FullProvenanceReader::FullProvenanceReader(RootTree* rootTree, DaqProvenanceHelper const* daqProvenanceHelper) :
         ProvenanceReaderBase(),
         rootTree_(rootTree),
         infoVector_(),
         pInfoVector_(&infoVector_),
         daqProvenanceHelper_(daqProvenanceHelper) {
  }

  void
  FullProvenanceReader::readProvenance(BranchMapper const& mapper) const {
    rootTree_->fillBranchEntryMeta(rootTree_->branchEntryInfoBranch(), pInfoVector_);
    setRefCoreStreamer(true);
    if(daqProvenanceHelper_) {
      for(auto const& info : infoVector_) {
        mapper.insertIntoSet(ProductProvenance(daqProvenanceHelper_->mapBranchID(info.branchID()),
                                               daqProvenanceHelper_->mapParentageID(info.parentageID())));
      }
    } else {
      for(auto const& info : infoVector_) {
        mapper.insertIntoSet(info);
      }
    }
  }

  class OldProvenanceReader : public ProvenanceReaderBase {
  public:
    explicit OldProvenanceReader(RootTree* rootTree, DaqProvenanceHelper const* daqProvenanceHelper);
    virtual ~OldProvenanceReader() {}
  private:
    virtual void readProvenance(BranchMapper const& mapper) const;
    RootTree* rootTree_;
    std::vector<EventEntryInfo> infoVector_;
    mutable std::vector<EventEntryInfo> *pInfoVector_;
    DaqProvenanceHelper const* daqProvenanceHelper_;
  };

  OldProvenanceReader::OldProvenanceReader(RootTree* rootTree, DaqProvenanceHelper const* daqProvenanceHelper) :
         ProvenanceReaderBase(),
         rootTree_(rootTree),
         infoVector_(),
         pInfoVector_(&infoVector_),
         daqProvenanceHelper_(daqProvenanceHelper) {
  }

  void
  OldProvenanceReader::readProvenance(BranchMapper const& mapper) const {
    rootTree_->branchEntryInfoBranch()->SetAddress(&pInfoVector_);
    roottree::getEntry(rootTree_->branchEntryInfoBranch(), rootTree_->entryNumber());
    setRefCoreStreamer(true);
    for(auto const& info : infoVector_) {
      EventEntryDescription eed;
      EntryDescriptionRegistry::instance()->getMapped(info.entryDescriptionID(), eed);
      Parentage parentage(eed.parents());
      if(daqProvenanceHelper_) {
        ProductProvenance entry(daqProvenanceHelper_->mapBranchID(info.branchID()),
                                daqProvenanceHelper_->mapParentageID(parentage.id()));
        mapper.insertIntoSet(entry);
      } else {
        ProductProvenance entry(info.branchID(), parentage.id());
        mapper.insertIntoSet(entry);
      }
    
    }
  }

  class DummyProvenanceReader : public ProvenanceReaderBase {
  public:
    DummyProvenanceReader();
    virtual ~DummyProvenanceReader() {}
  private:
    virtual void readProvenance(BranchMapper const& mapper) const;
  };

  DummyProvenanceReader::DummyProvenanceReader() :
      ProvenanceReaderBase() {
  }

  void
  DummyProvenanceReader::readProvenance(BranchMapper const&) const {
    // Not providing parentage!!!
  }

  std::unique_ptr<ProvenanceReaderBase>
  MakeDummyProvenanceReader::makeReader(RootTree&, DaqProvenanceHelper const*) const {
     return std::unique_ptr<ProvenanceReaderBase>(new DummyProvenanceReader);
  }

  std::unique_ptr<ProvenanceReaderBase>
  MakeOldProvenanceReader::makeReader(RootTree& rootTree, DaqProvenanceHelper const* daqProvenanceHelper) const {
    return std::unique_ptr<ProvenanceReaderBase>(new OldProvenanceReader(&rootTree, daqProvenanceHelper));
  }

  std::unique_ptr<ProvenanceReaderBase>
  MakeFullProvenanceReader::makeReader(RootTree& rootTree, DaqProvenanceHelper const* daqProvenanceHelper) const {
    return std::unique_ptr<ProvenanceReaderBase>(new FullProvenanceReader(&rootTree, daqProvenanceHelper));
  }

  std::unique_ptr<ProvenanceReaderBase>
  MakeReducedProvenanceReader::makeReader(RootTree& rootTree, DaqProvenanceHelper const* daqProvenanceHelper) const {
    return std::unique_ptr<ProvenanceReaderBase>(new ReducedProvenanceReader(&rootTree, parentageIDLookup_, daqProvenanceHelper));
  }
}
