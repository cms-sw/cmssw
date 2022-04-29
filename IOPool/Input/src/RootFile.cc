/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootFile.h"
#include "DuplicateChecker.h"
#include "InputFile.h"
#include "ProvenanceAdaptor.h"
#include "RunHelper.h"

#include "DataFormats/Common/interface/setIsMergeable.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/StoredMergeableRunProductMetadata.h"
#include "DataFormats/Provenance/interface/StoredProcessBlockHelper.h"
#include "DataFormats/Provenance/interface/StoredProductProvenance.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/Common/interface/ProcessBlockHelper.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/MergeableRunProductMetadata.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"
#include "FWCore/Sources/interface/DaqProvenanceHelper.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/ReleaseVersion.h"
#include "FWCore/Utilities/interface/stemFromPath.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "IOPool/Common/interface/getWrapperBasePtr.h"

#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

//used for backward compatibility
#include "DataFormats/Provenance/interface/EventAux.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"
#include "DataFormats/Provenance/interface/RunAux.h"
#include "FWCore/ParameterSet/interface/ParameterSetConverter.h"

#include "Rtypes.h"
#include "TClass.h"
#include "TString.h"
#include "TTree.h"
#include "TTreeCache.h"

#include <algorithm>
#include <cassert>
#include <list>

namespace edm {

  // Algorithm classes for making ProvenanceReader:
  class MakeDummyProvenanceReader : public MakeProvenanceReader {
  public:
    std::unique_ptr<ProvenanceReaderBase> makeReader(RootTree& eventTree,
                                                     DaqProvenanceHelper const* daqProvenanceHelper) const override;
  };
  class MakeOldProvenanceReader : public MakeProvenanceReader {
  public:
    MakeOldProvenanceReader(std::unique_ptr<EntryDescriptionMap>&& entryDescriptionMap)
        : MakeProvenanceReader(), entryDescriptionMap_(std::move(entryDescriptionMap)) {}
    std::unique_ptr<ProvenanceReaderBase> makeReader(RootTree& eventTree,
                                                     DaqProvenanceHelper const* daqProvenanceHelper) const override;

  private:
    edm::propagate_const<std::unique_ptr<EntryDescriptionMap>> entryDescriptionMap_;
  };
  class MakeFullProvenanceReader : public MakeProvenanceReader {
  public:
    std::unique_ptr<ProvenanceReaderBase> makeReader(RootTree& eventTree,
                                                     DaqProvenanceHelper const* daqProvenanceHelper) const override;
  };
  class MakeReducedProvenanceReader : public MakeProvenanceReader {
  public:
    MakeReducedProvenanceReader(std::vector<ParentageID> const& parentageIDLookup)
        : parentageIDLookup_(parentageIDLookup) {}
    std::unique_ptr<ProvenanceReaderBase> makeReader(RootTree& eventTree,
                                                     DaqProvenanceHelper const* daqProvenanceHelper) const override;

  private:
    std::vector<ParentageID> const& parentageIDLookup_;
  };

  namespace {
    void checkReleaseVersion(std::vector<ProcessHistory> processHistoryVector, std::string const& fileName) {
      std::string releaseVersion = getReleaseVersion();
      releaseversion::DecomposedReleaseVersion currentRelease(releaseVersion);
      for (auto const& ph : processHistoryVector) {
        for (auto const& pc : ph) {
          if (releaseversion::isEarlierRelease(currentRelease, pc.releaseVersion())) {
            throw Exception(errors::FormatIncompatibility)
                << "The release you are using, " << getReleaseVersion() << " , predates\n"
                << "a release (" << pc.releaseVersion() << ") used in writing the input file, " << fileName << ".\n"
                << "Forward compatibility cannot be supported.\n";
          }
        }
      }
    }
  }  // namespace

  // This is a helper class for IndexIntoFile.
  class RootFileEventFinder : public IndexIntoFile::EventFinder {
  public:
    explicit RootFileEventFinder(RootTree& eventTree) : eventTree_(eventTree) {}
    ~RootFileEventFinder() override {}

    EventNumber_t getEventNumberOfEntry(roottree::EntryNumber entry) const override {
      roottree::EntryNumber saveEntry = eventTree_.entryNumber();
      eventTree_.setEntryNumber(entry);
      EventAuxiliary eventAux;
      EventAuxiliary* pEvAux = &eventAux;
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
                     std::shared_ptr<InputFile> filePtr,
                     std::shared_ptr<EventSkipperByID> eventSkipperByID,
                     bool skipAnyEvents,
                     int remainingEvents,
                     int remainingLumis,
                     unsigned int nStreams,
                     unsigned int treeCacheSize,
                     int treeMaxVirtualSize,
                     InputSource::ProcessingMode processingMode,
                     RunHelperBase* runHelper,
                     bool noRunLumiSort,
                     bool noEventSort,
                     ProductSelectorRules const& productSelectorRules,
                     InputType inputType,
                     std::shared_ptr<BranchIDListHelper> branchIDListHelper,
                     ProcessBlockHelper* processBlockHelper,
                     std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper,
                     std::vector<BranchID> const* associationsFromSecondary,
                     std::shared_ptr<DuplicateChecker> duplicateChecker,
                     bool dropDescendants,
                     ProcessHistoryRegistry& processHistoryRegistry,
                     std::vector<std::shared_ptr<IndexIntoFile>> const& indexesIntoFiles,
                     std::vector<std::shared_ptr<IndexIntoFile>>::size_type currentIndexIntoFile,
                     std::vector<ProcessHistoryID>& orderedProcessHistoryIDs,
                     bool bypassVersionCheck,
                     bool labelRawDataLikeMC,
                     bool usingGoToEvent,
                     bool enablePrefetching,
                     bool enforceGUIDInFileName)
      : file_(fileName),
        logicalFile_(logicalFileName),
        processConfiguration_(processConfiguration),
        processHistoryRegistry_(&processHistoryRegistry),
        filePtr_(filePtr),
        eventSkipperByID_(eventSkipperByID),
        fileFormatVersion_(),
        fid_(),
        indexIntoFileSharedPtr_(new IndexIntoFile),
        indexIntoFile_(*indexIntoFileSharedPtr_),
        orderedProcessHistoryIDs_(orderedProcessHistoryIDs),
        indexIntoFileBegin_(indexIntoFile_.begin(
            noRunLumiSort ? IndexIntoFile::entryOrder
                          : (noEventSort ? IndexIntoFile::firstAppearanceOrder : IndexIntoFile::numericalOrder))),
        indexIntoFileEnd_(indexIntoFileBegin_),
        indexIntoFileIter_(indexIntoFileBegin_),
        storedMergeableRunProductMetadata_((inputType == InputType::Primary) ? new StoredMergeableRunProductMetadata
                                                                             : nullptr),
        eventProcessHistoryIDs_(),
        eventProcessHistoryIter_(eventProcessHistoryIDs_.begin()),
        savedRunAuxiliary_(),
        skipAnyEvents_(skipAnyEvents),
        noRunLumiSort_(noRunLumiSort),
        noEventSort_(noEventSort),
        enforceGUIDInFileName_(enforceGUIDInFileName),
        whyNotFastClonable_(0),
        hasNewlyDroppedBranch_(),
        branchListIndexesUnchanged_(false),
        eventAuxCache_(),
        eventTree_(filePtr,
                   InEvent,
                   nStreams,
                   treeMaxVirtualSize,
                   treeCacheSize,
                   roottree::defaultLearningEntries,
                   enablePrefetching,
                   inputType),
        lumiTree_(filePtr,
                  InLumi,
                  1,
                  treeMaxVirtualSize,
                  roottree::defaultNonEventCacheSize,
                  roottree::defaultNonEventLearningEntries,
                  enablePrefetching,
                  inputType),
        runTree_(filePtr,
                 InRun,
                 1,
                 treeMaxVirtualSize,
                 roottree::defaultNonEventCacheSize,
                 roottree::defaultNonEventLearningEntries,
                 enablePrefetching,
                 inputType),
        treePointers_(),
        lastEventEntryNumberRead_(IndexIntoFile::invalidEntry),
        productRegistry_(),
        branchIDLists_(),
        branchIDListHelper_(branchIDListHelper),
        processBlockHelper_(processBlockHelper),
        fileThinnedAssociationsHelper_(),
        thinnedAssociationsHelper_(thinnedAssociationsHelper),
        processingMode_(processingMode),
        runHelper_(runHelper),
        newBranchToOldBranch_(),
        eventHistoryTree_(nullptr),
        eventToProcessBlockIndexesBranch_(
            inputType == InputType::Primary
                ? eventTree_.tree()->GetBranch(poolNames::eventToProcessBlockIndexesBranchName().c_str())
                : nullptr),
        history_(),
        branchChildren_(new BranchChildren),
        duplicateChecker_(duplicateChecker),
        provenanceAdaptor_(),
        provenanceReaderMaker_(),
        eventProductProvenanceRetrievers_(),
        parentageIDLookup_(),
        daqProvenanceHelper_(),
        edProductClass_(TypeWithDict::byName("edm::WrapperBase").getClass()),
        inputType_(inputType) {
    hasNewlyDroppedBranch_.fill(false);

    treePointers_.resize(3);
    treePointers_[InEvent] = &eventTree_;
    treePointers_[InLumi] = &lumiTree_;
    treePointers_[InRun] = &runTree_;

    // Read the metadata tree.
    // We use a smart pointer so the tree will be deleted after use, and not kept for the life of the file.
    std::unique_ptr<TTree> metaDataTree(dynamic_cast<TTree*>(filePtr_->Get(poolNames::metaDataTreeName().c_str())));
    if (nullptr == metaDataTree.get()) {
      throw Exception(errors::FileReadError)
          << "Could not find tree " << poolNames::metaDataTreeName() << " in the input file.\n";
    }

    // To keep things simple, we just read in every possible branch that exists.
    // We don't pay attention to which branches exist in which file format versions

    FileFormatVersion* fftPtr = &fileFormatVersion_;
    if (metaDataTree->FindBranch(poolNames::fileFormatVersionBranchName().c_str()) != nullptr) {
      TBranch* fft = metaDataTree->GetBranch(poolNames::fileFormatVersionBranchName().c_str());
      fft->SetAddress(&fftPtr);
      roottree::getEntry(fft, 0);
      metaDataTree->SetBranchAddress(poolNames::fileFormatVersionBranchName().c_str(), &fftPtr);
    }

    FileID* fidPtr = &fid_;
    if (metaDataTree->FindBranch(poolNames::fileIdentifierBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::fileIdentifierBranchName().c_str(), &fidPtr);
    }

    IndexIntoFile* iifPtr = &indexIntoFile_;
    if (metaDataTree->FindBranch(poolNames::indexIntoFileBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::indexIntoFileBranchName().c_str(), &iifPtr);
    }

    storedProcessBlockHelper_ = std::make_unique<StoredProcessBlockHelper>();
    StoredProcessBlockHelper& storedProcessBlockHelper = *storedProcessBlockHelper_;
    StoredProcessBlockHelper* pStoredProcessBlockHelper = storedProcessBlockHelper_.get();
    if (inputType == InputType::Primary) {
      if (metaDataTree->FindBranch(poolNames::processBlockHelperBranchName().c_str()) != nullptr) {
        metaDataTree->SetBranchAddress(poolNames::processBlockHelperBranchName().c_str(), &pStoredProcessBlockHelper);
      }
    }

    StoredMergeableRunProductMetadata* smrc = nullptr;
    if (inputType == InputType::Primary) {
      smrc = &*storedMergeableRunProductMetadata_;
      if (metaDataTree->FindBranch(poolNames::mergeableRunProductMetadataBranchName().c_str()) != nullptr) {
        metaDataTree->SetBranchAddress(poolNames::mergeableRunProductMetadataBranchName().c_str(), &smrc);
      }
    }

    // Need to read to a temporary registry so we can do a translation of the BranchKeys.
    // This preserves backward compatibility against friendly class name algorithm changes.
    ProductRegistry inputProdDescReg;
    ProductRegistry* ppReg = &inputProdDescReg;
    metaDataTree->SetBranchAddress(poolNames::productDescriptionBranchName().c_str(), (&ppReg));

    using PsetMap = std::map<ParameterSetID, ParameterSetBlob>;
    PsetMap psetMap;
    PsetMap* psetMapPtr = &psetMap;
    if (metaDataTree->FindBranch(poolNames::parameterSetMapBranchName().c_str()) != nullptr) {
      //backward compatibility
      assert(!fileFormatVersion().parameterSetsTree());
      metaDataTree->SetBranchAddress(poolNames::parameterSetMapBranchName().c_str(), &psetMapPtr);
    } else {
      assert(fileFormatVersion().parameterSetsTree());
      // We use a smart pointer so the tree will be deleted after use, and not kept for the life of the file.
      std::unique_ptr<TTree> psetTree(dynamic_cast<TTree*>(filePtr_->Get(poolNames::parameterSetsTreeName().c_str())));
      if (nullptr == psetTree.get()) {
        throw Exception(errors::FileReadError)
            << "Could not find tree " << poolNames::parameterSetsTreeName() << " in the input file.\n";
      }

      using IdToBlobs = std::pair<ParameterSetID, ParameterSetBlob>;
      IdToBlobs idToBlob;
      IdToBlobs* pIdToBlob = &idToBlob;
      psetTree->SetBranchAddress(poolNames::idToParameterSetBlobsBranchName().c_str(), &pIdToBlob);

      std::unique_ptr<TTreeCache> psetTreeCache =
          roottree::trainCache(psetTree.get(), *filePtr_, roottree::defaultNonEventCacheSize, "*");
      psetTreeCache->SetEnablePrefetching(false);
      filePtr_->SetCacheRead(psetTreeCache.get());
      for (Long64_t i = 0; i != psetTree->GetEntries(); ++i) {
        psetTree->GetEntry(i);
        psetMap.insert(idToBlob);
      }
      filePtr_->SetCacheRead(nullptr);
    }

    // backward compatibility
    ProcessHistoryRegistry::collection_type pHistMap;
    ProcessHistoryRegistry::collection_type* pHistMapPtr = &pHistMap;
    if (metaDataTree->FindBranch(poolNames::processHistoryMapBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::processHistoryMapBranchName().c_str(), &pHistMapPtr);
    }

    ProcessHistoryRegistry::vector_type pHistVector;
    ProcessHistoryRegistry::vector_type* pHistVectorPtr = &pHistVector;
    if (metaDataTree->FindBranch(poolNames::processHistoryBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::processHistoryBranchName().c_str(), &pHistVectorPtr);
    }

    // backward compatibility
    ProcessConfigurationVector processConfigurations;
    ProcessConfigurationVector* procConfigVectorPtr = &processConfigurations;
    if (metaDataTree->FindBranch(poolNames::processConfigurationBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::processConfigurationBranchName().c_str(), &procConfigVectorPtr);
    }

    auto branchIDListsAPtr = std::make_unique<BranchIDLists>();
    BranchIDLists* branchIDListsPtr = branchIDListsAPtr.get();
    if (metaDataTree->FindBranch(poolNames::branchIDListBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::branchIDListBranchName().c_str(), &branchIDListsPtr);
    }

    ThinnedAssociationsHelper* thinnedAssociationsHelperPtr;  // must remain in scope through getEntry()
    if (inputType != InputType::SecondarySource) {
      fileThinnedAssociationsHelper_ =
          std::make_unique<ThinnedAssociationsHelper>();  // propagate_const<T> has no reset() function
      thinnedAssociationsHelperPtr = fileThinnedAssociationsHelper_.get();
      if (metaDataTree->FindBranch(poolNames::thinnedAssociationsHelperBranchName().c_str()) != nullptr) {
        metaDataTree->SetBranchAddress(poolNames::thinnedAssociationsHelperBranchName().c_str(),
                                       &thinnedAssociationsHelperPtr);
      }
    }

    BranchChildren* branchChildrenBuffer = branchChildren_.get();
    if (metaDataTree->FindBranch(poolNames::productDependenciesBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::productDependenciesBranchName().c_str(), &branchChildrenBuffer);
    }

    // backward compatibility
    std::vector<EventProcessHistoryID>* eventHistoryIDsPtr = &eventProcessHistoryIDs_;
    if (metaDataTree->FindBranch(poolNames::eventHistoryBranchName().c_str()) != nullptr) {
      metaDataTree->SetBranchAddress(poolNames::eventHistoryBranchName().c_str(), &eventHistoryIDsPtr);
    }

    if (metaDataTree->FindBranch(poolNames::moduleDescriptionMapBranchName().c_str()) != nullptr) {
      if (metaDataTree->GetBranch(poolNames::moduleDescriptionMapBranchName().c_str())->GetSplitLevel() != 0) {
        metaDataTree->SetBranchStatus((poolNames::moduleDescriptionMapBranchName() + ".*").c_str(), false);
      } else {
        metaDataTree->SetBranchStatus(poolNames::moduleDescriptionMapBranchName().c_str(), false);
      }
    }

    // Here we read the metadata tree
    roottree::getEntry(metaDataTree.get(), 0);

    eventProcessHistoryIter_ = eventProcessHistoryIDs_.begin();

    // Here we read the event history tree, if we have one.
    readEventHistoryTree();

    ParameterSetConverter::ParameterSetIdConverter psetIdConverter;
    if (!fileFormatVersion().triggerPathsTracked()) {
      ParameterSetConverter converter(psetMap, psetIdConverter, fileFormatVersion().parameterSetsByReference());
    } else {
      // Merge into the parameter set registry.
      pset::Registry& psetRegistry = *pset::Registry::instance();
      for (auto const& psetEntry : psetMap) {
        ParameterSet pset(psetEntry.second.pset());
        pset.setID(psetEntry.first);
        // For thread safety, don't update global registries when a secondary source opens a file.
        if (inputType != InputType::SecondarySource) {
          psetRegistry.insertMapped(pset);
        }
      }
    }
    if (!fileFormatVersion().splitProductIDs()) {
      // Old provenance format input file.  Create a provenance adaptor.
      // propagate_const<T> has no reset() function
      provenanceAdaptor_ = std::make_unique<ProvenanceAdaptor>(
          inputProdDescReg, pHistMap, pHistVector, processConfigurations, psetIdConverter, true);
      // Fill in the branchIDLists branch from the provenance adaptor
      branchIDLists_ = provenanceAdaptor_->branchIDLists();
    } else {
      if (!fileFormatVersion().triggerPathsTracked()) {
        // New provenance format, but change in ParameterSet Format. Create a provenance adaptor.
        // propagate_const<T> has no reset() function
        provenanceAdaptor_ = std::make_unique<ProvenanceAdaptor>(
            inputProdDescReg, pHistMap, pHistVector, processConfigurations, psetIdConverter, false);
      }
      // New provenance format input file. The branchIDLists branch was read directly from the input file.
      if (metaDataTree->FindBranch(poolNames::branchIDListBranchName().c_str()) == nullptr) {
        throw Exception(errors::EventCorruption) << "Failed to find branchIDLists branch in metaData tree.\n";
      }
      branchIDLists_.reset(branchIDListsAPtr.release());
    }

    if (fileFormatVersion().hasThinnedAssociations()) {
      if (metaDataTree->FindBranch(poolNames::thinnedAssociationsHelperBranchName().c_str()) == nullptr) {
        throw Exception(errors::EventCorruption)
            << "Failed to find thinnedAssociationsHelper branch in metaData tree.\n";
      }
    }

    if (!bypassVersionCheck) {
      checkReleaseVersion(pHistVector, file());
    }

    if (labelRawDataLikeMC) {
      std::string const rawData("FEDRawDataCollection");
      std::string const source("source");
      ProductRegistry::ProductList& pList = inputProdDescReg.productListUpdator();
      BranchKey finder(rawData, source, "", "");
      ProductRegistry::ProductList::iterator it = pList.lower_bound(finder);
      if (it != pList.end() && it->first.friendlyClassName() == rawData && it->first.moduleLabel() == source) {
        // We found raw data with a module label of source.
        // We need to change the module label and process name.
        // Create helper.
        it->second.init();
        // propagate_const<T> has no reset() function
        daqProvenanceHelper_ = std::make_unique<DaqProvenanceHelper>(it->second.unwrappedTypeID());
        // Create the new branch description
        BranchDescription const& newBD = daqProvenanceHelper_->branchDescription();
        // Save info from the old and new branch descriptions
        daqProvenanceHelper_->saveInfo(it->second, newBD);
        // Map the new branch name to the old branch name.
        newBranchToOldBranch_.insert(std::make_pair(newBD.branchName(), it->second.branchName()));
        // Remove the old branch description from the product Registry.
        pList.erase(it);
        // Check that there was only one.
        it = pList.lower_bound(finder);
        assert(!(it != pList.end() && it->first.friendlyClassName() == rawData && it->first.moduleLabel() == source));
        // Insert the new branch description into the product registry.
        inputProdDescReg.copyProduct(newBD);
        // Fix up other per file metadata.
        daqProvenanceHelper_->fixMetaData(processConfigurations, pHistVector);
        daqProvenanceHelper_->fixMetaData(*branchIDLists_);
        daqProvenanceHelper_->fixMetaData(*branchChildren_);
      }
    }

    for (auto const& history : pHistVector) {
      processHistoryRegistry.registerProcessHistory(history);
    }

    eventTree_.trainCache(BranchTypeToAuxiliaryBranchName(InEvent).c_str());

    // Update the branch id info. This has to be done before validateFile since
    // depending on the file format, the branchIDListHelper_ may have its fixBranchListIndexes call made
    if (inputType == InputType::Primary) {
      branchListIndexesUnchanged_ = branchIDListHelper_->updateFromInput(*branchIDLists_);
    }

    validateFile(inputType, usingGoToEvent);

    // Here, we make the class that will make the ProvenanceReader
    // It reads whatever trees it needs.
    // propagate_const<T> has no reset() function
    provenanceReaderMaker_ = std::unique_ptr<MakeProvenanceReader>(makeProvenanceReaderMaker(inputType).release());

    // Merge into the hashed registries.
    if (eventSkipperByID_ && eventSkipperByID_->somethingToSkip()) {
      whyNotFastClonable_ += FileBlock::EventsOrLumisSelectedByID;
    }

    initializeDuplicateChecker(indexesIntoFiles, currentIndexIntoFile);
    indexIntoFileIter_ = indexIntoFileBegin_ = indexIntoFile_.begin(
        noRunLumiSort ? IndexIntoFile::entryOrder
                      : (noEventSort ? IndexIntoFile::firstAppearanceOrder : IndexIntoFile::numericalOrder));
    indexIntoFileEnd_ = indexIntoFile_.end(
        noRunLumiSort ? IndexIntoFile::entryOrder
                      : (noEventSort ? IndexIntoFile::firstAppearanceOrder : IndexIntoFile::numericalOrder));
    runHelper_->setForcedRunOffset(indexIntoFileBegin_ == indexIntoFileEnd_ ? 1 : indexIntoFileBegin_.run());
    eventProcessHistoryIter_ = eventProcessHistoryIDs_.begin();

    makeProcessBlockRootTrees(filePtr, treeMaxVirtualSize, enablePrefetching, inputType, storedProcessBlockHelper);

    setPresenceInProductRegistry(inputProdDescReg, storedProcessBlockHelper);

    auto newReg = std::make_unique<ProductRegistry>();

    // Do the translation from the old registry to the new one
    {
      ProductRegistry::ProductList const& prodList = inputProdDescReg.productList();
      for (auto const& product : prodList) {
        BranchDescription const& prod = product.second;
        std::string newFriendlyName = friendlyname::friendlyName(prod.className());
        if (newFriendlyName == prod.friendlyClassName()) {
          newReg->copyProduct(prod);
        } else {
          if (fileFormatVersion().splitProductIDs()) {
            throw Exception(errors::UnimplementedFeature)
                << "Cannot change friendly class name algorithm without more development work\n"
                << "to update BranchIDLists and ThinnedAssociationsHelper.  Contact the framework group.\n";
          }
          BranchDescription newBD(prod);
          newBD.updateFriendlyClassName();
          newReg->copyProduct(newBD);
          newBranchToOldBranch_.insert(std::make_pair(newBD.branchName(), prod.branchName()));
        }
      }

      dropOnInputAndReorder(
          *newReg, productSelectorRules, dropDescendants, inputType, storedProcessBlockHelper, processBlockHelper);

      if (inputType == InputType::SecondaryFile) {
        thinnedAssociationsHelper->updateFromSecondaryInput(*fileThinnedAssociationsHelper_,
                                                            *associationsFromSecondary);
      } else if (inputType == InputType::Primary) {
        processBlockHelper->initializeFromPrimaryInput(storedProcessBlockHelper);
        thinnedAssociationsHelper->updateFromPrimaryInput(*fileThinnedAssociationsHelper_);
      }

      if (inputType == InputType::Primary) {
        for (auto& product : newReg->productListUpdator()) {
          setIsMergeable(product.second);
        }
      }
      //inform system we want to use DelayedReader
      for (auto& product : newReg->productListUpdator()) {
        product.second.setOnDemand(true);
      }

      for (auto& processBlockTree : processBlockTrees_) {
        treePointers_.push_back(processBlockTree.get());
      }

      // freeze the product registry
      newReg->setFrozen(inputType != InputType::Primary);
      productRegistry_.reset(newReg.release());
    }

    // Set up information from the product registry.
    ProductRegistry::ProductList const& prodList = productRegistry()->productList();

    {
      std::vector<size_t> nBranches(treePointers_.size(), 0);
      for (auto const& product : prodList) {
        if (product.second.branchType() == InProcess) {
          std::vector<std::string> const& processes = storedProcessBlockHelper.processesWithProcessBlockProducts();
          auto it = std::find(processes.begin(), processes.end(), product.second.processName());
          if (it != processes.end()) {
            auto index = std::distance(processes.begin(), it);
            ++nBranches[numberOfRunLumiEventProductTrees + index];
          }
        } else {
          ++nBranches[product.second.branchType()];
        }
      }

      int i = 0;
      for (auto& t : treePointers_) {
        t->numberOfBranchesToAdd(nBranches[i]);
        ++i;
      }
    }
    for (auto const& product : prodList) {
      BranchDescription const& prod = product.second;
      if (prod.branchType() == InProcess) {
        std::vector<std::string> const& processes = storedProcessBlockHelper.processesWithProcessBlockProducts();
        auto it = std::find(processes.begin(), processes.end(), prod.processName());
        if (it != processes.end()) {
          auto index = std::distance(processes.begin(), it);
          treePointers_[numberOfRunLumiEventProductTrees + index]->addBranch(prod,
                                                                             newBranchToOldBranch(prod.branchName()));
        }
      } else {
        treePointers_[prod.branchType()]->addBranch(prod, newBranchToOldBranch(prod.branchName()));
      }
    }

    // Determine if this file is fast clonable.
    setIfFastClonable(remainingEvents, remainingLumis);

    // We are done with our initial reading of EventAuxiliary.
    indexIntoFile_.doneFileInitialization();

    // Tell the event tree to begin training at the next read.
    eventTree_.resetTraining();

    // Train the run and lumi trees.
    runTree_.trainCache("*");
    lumiTree_.trainCache("*");
    for (auto& processBlockTree : processBlockTrees_) {
      processBlockTree->trainCache("*");
    }
  }

  RootFile::~RootFile() {}

  void RootFile::readEntryDescriptionTree(EntryDescriptionMap& entryDescriptionMap, InputType inputType) {
    // Called only for old format files.
    // We use a smart pointer so the tree will be deleted after use, and not kept for the life of the file.
    std::unique_ptr<TTree> entryDescriptionTree(
        dynamic_cast<TTree*>(filePtr_->Get(poolNames::entryDescriptionTreeName().c_str())));
    if (nullptr == entryDescriptionTree.get()) {
      throw Exception(errors::FileReadError)
          << "Could not find tree " << poolNames::entryDescriptionTreeName() << " in the input file.\n";
    }

    EntryDescriptionID idBuffer;
    EntryDescriptionID* pidBuffer = &idBuffer;
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionIDBranchName().c_str(), &pidBuffer);

    EventEntryDescription entryDescriptionBuffer;
    EventEntryDescription* pEntryDescriptionBuffer = &entryDescriptionBuffer;
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionBranchName().c_str(), &pEntryDescriptionBuffer);

    // Fill in the parentage registry.
    ParentageRegistry& registry = *ParentageRegistry::instance();

    for (Long64_t i = 0, numEntries = entryDescriptionTree->GetEntries(); i < numEntries; ++i) {
      roottree::getEntry(entryDescriptionTree.get(), i);
      if (idBuffer != entryDescriptionBuffer.id()) {
        throw Exception(errors::EventCorruption) << "Corruption of EntryDescription tree detected.\n";
      }
      entryDescriptionMap.insert(std::make_pair(entryDescriptionBuffer.id(), entryDescriptionBuffer));
      Parentage parents;
      parents.setParents(entryDescriptionBuffer.parents());
      if (daqProvenanceHelper_) {
        ParentageID const oldID = parents.id();
        daqProvenanceHelper_->fixMetaData(parents.parentsForUpdate());
        ParentageID newID = parents.id();
        if (newID != oldID) {
          daqProvenanceHelper_->setOldParentageIDToNew(oldID, newID);
        }
      }
      // For thread safety, don't update global registries when a secondary source opens a file.
      if (inputType != InputType::SecondarySource) {
        registry.insertMapped(parents);
      }
    }
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionIDBranchName().c_str(), nullptr);
    entryDescriptionTree->SetBranchAddress(poolNames::entryDescriptionBranchName().c_str(), nullptr);
  }

  void RootFile::readParentageTree(InputType inputType) {
    // New format file
    // We use a smart pointer so the tree will be deleted after use, and not kept for the life of the file.
    std::unique_ptr<TTree> parentageTree(dynamic_cast<TTree*>(filePtr_->Get(poolNames::parentageTreeName().c_str())));
    if (nullptr == parentageTree.get()) {
      throw Exception(errors::FileReadError)
          << "Could not find tree " << poolNames::parentageTreeName() << " in the input file.\n";
    }

    Parentage parents;
    Parentage* pParentageBuffer = &parents;
    parentageTree->SetBranchAddress(poolNames::parentageBranchName().c_str(), &pParentageBuffer);

    ParentageRegistry& registry = *ParentageRegistry::instance();

    parentageIDLookup_.reserve(parentageTree->GetEntries());
    for (Long64_t i = 0, numEntries = parentageTree->GetEntries(); i < numEntries; ++i) {
      roottree::getEntry(parentageTree.get(), i);
      if (daqProvenanceHelper_) {
        ParentageID const oldID = parents.id();
        daqProvenanceHelper_->fixMetaData(parents.parentsForUpdate());
        ParentageID newID = parents.id();
        if (newID != oldID) {
          daqProvenanceHelper_->setOldParentageIDToNew(oldID, newID);
        }
      }
      // For thread safety, don't update global registries when a secondary source opens a file.
      if (inputType != InputType::SecondarySource) {
        registry.insertMapped(parents);
      }
      parentageIDLookup_.push_back(parents.id());
    }
    parentageTree->SetBranchAddress(poolNames::parentageBranchName().c_str(), nullptr);
  }

  void RootFile::setIfFastClonable(int remainingEvents, int remainingLumis) {
    if (fileFormatVersion().noMetaDataTrees() and !fileFormatVersion().storedProductProvenanceUsed()) {
      //we must avoid copying the old branch which stored the per product per event provenance
      whyNotFastClonable_ += FileBlock::FileTooOld;
      return;
    }
    if (!fileFormatVersion().splitProductIDs()) {
      whyNotFastClonable_ += FileBlock::FileTooOld;
      return;
    }
    if (processingMode_ != InputSource::RunsLumisAndEvents) {
      whyNotFastClonable_ += FileBlock::NotProcessingEvents;
      return;
    }
    // Find entry for first event in file
    IndexIntoFile::IndexIntoFileItr it = indexIntoFileBegin_;
    while (it != indexIntoFileEnd_ && it.getEntryType() != IndexIntoFile::kEvent) {
      ++it;
    }
    if (it == indexIntoFileEnd_) {
      whyNotFastClonable_ += FileBlock::NoEventsInFile;
      return;
    }

    // From here on, record all reasons we can't fast clone.
    IndexIntoFile::SortOrder sortOrder =
        (noRunLumiSort_ ? IndexIntoFile::entryOrder
                        : (noEventSort_ ? IndexIntoFile::firstAppearanceOrder : IndexIntoFile::numericalOrder));
    if (!indexIntoFile_.iterationWillBeInEntryOrder(sortOrder)) {
      whyNotFastClonable_ += (noEventSort_ ? FileBlock::RunOrLumiNotContiguous : FileBlock::EventsToBeSorted);
    }
    if (skipAnyEvents_) {
      whyNotFastClonable_ += FileBlock::InitialEventsSkipped;
    }
    if (remainingEvents >= 0 && eventTree_.entries() > remainingEvents) {
      whyNotFastClonable_ += FileBlock::MaxEventsTooSmall;
    }
    if (remainingLumis >= 0 && lumiTree_.entries() > remainingLumis) {
      whyNotFastClonable_ += FileBlock::MaxLumisTooSmall;
    }
    if (duplicateChecker_ && !duplicateChecker_->checkDisabled() && !duplicateChecker_->noDuplicatesInFile()) {
      whyNotFastClonable_ += FileBlock::DuplicateEventsRemoved;
    }
  }

  std::shared_ptr<FileBlock> RootFile::createFileBlock() {
    std::vector<TTree*> processBlockTrees;
    std::vector<std::string> processesWithProcessBlockTrees;
    processBlockTrees.reserve(processBlockTrees_.size());
    processesWithProcessBlockTrees.reserve(processBlockTrees_.size());
    for (auto& processBlockTree : processBlockTrees_) {
      processBlockTrees.push_back(processBlockTree->tree());
      processesWithProcessBlockTrees.push_back(processBlockTree->processName());
    }
    return std::make_shared<FileBlock>(fileFormatVersion(),
                                       eventTree_.tree(),
                                       eventTree_.metaTree(),
                                       lumiTree_.tree(),
                                       lumiTree_.metaTree(),
                                       runTree_.tree(),
                                       runTree_.metaTree(),
                                       std::move(processBlockTrees),
                                       std::move(processesWithProcessBlockTrees),
                                       whyNotFastClonable(),
                                       hasNewlyDroppedBranch(),
                                       file_,
                                       branchListIndexesUnchanged(),
                                       modifiedIDs(),
                                       branchChildren());
  }

  void RootFile::updateFileBlock(FileBlock& fileBlock) {
    std::vector<TTree*> processBlockTrees;
    std::vector<std::string> processesWithProcessBlockTrees;
    processBlockTrees.reserve(processBlockTrees_.size());
    processesWithProcessBlockTrees.reserve(processBlockTrees_.size());
    for (auto& processBlockTree : processBlockTrees_) {
      processBlockTrees.push_back(processBlockTree->tree());
      processesWithProcessBlockTrees.push_back(processBlockTree->processName());
    }
    fileBlock.updateTTreePointers(eventTree_.tree(),
                                  eventTree_.metaTree(),
                                  lumiTree_.tree(),
                                  lumiTree_.metaTree(),
                                  runTree_.tree(),
                                  runTree_.metaTree(),
                                  std::move(processBlockTrees),
                                  std::move(processesWithProcessBlockTrees));
  }

  std::string const& RootFile::newBranchToOldBranch(std::string const& newBranch) const {
    std::map<std::string, std::string>::const_iterator it = newBranchToOldBranch_.find(newBranch);
    if (it != newBranchToOldBranch_.end()) {
      return it->second;
    }
    return newBranch;
  }

  IndexIntoFile::IndexIntoFileItr RootFile::indexIntoFileIter() const { return indexIntoFileIter_; }

  void RootFile::setPosition(IndexIntoFile::IndexIntoFileItr const& position) {
    indexIntoFileIter_.copyPosition(position);
  }

  void RootFile::initAssociationsFromSecondary(std::vector<BranchID> const& associationsFromSecondary) {
    thinnedAssociationsHelper_->initAssociationsFromSecondary(associationsFromSecondary,
                                                              *fileThinnedAssociationsHelper_);
  }

  bool RootFile::skipThisEntry() {
    if (indexIntoFileIter_ == indexIntoFileEnd_) {
      return false;
    }

    if (eventSkipperByID_ && eventSkipperByID_->somethingToSkip()) {
      // See first if the entire lumi or run is skipped, so we won't have to read the event Auxiliary in that case.
      if (eventSkipperByID_->skipIt(indexIntoFileIter_.run(), indexIntoFileIter_.lumi(), 0U)) {
        return true;
      }

      // The Lumi is not skipped.  If this is an event, see if the event is skipped.
      if (indexIntoFileIter_.getEntryType() == IndexIntoFile::kEvent) {
        auto eventAux = fillEventAuxiliary(indexIntoFileIter_.entry());
        if (eventSkipperByID_->skipIt(indexIntoFileIter_.run(), indexIntoFileIter_.lumi(), eventAux.id().event())) {
          return true;
        }
      }

      // Skip runs with no lumis if either lumisToSkip or lumisToProcess have been set to select lumis
      if (indexIntoFileIter_.getEntryType() == IndexIntoFile::kRun && eventSkipperByID_->skippingLumis()) {
        // There are no lumis in this run, not even ones we will skip
        if (indexIntoFileIter_.peekAheadAtLumi() == IndexIntoFile::invalidLumi) {
          return true;
        }
        // If we get here there are lumis in the run, check to see if we are skipping all of them
        do {
          if (!eventSkipperByID_->skipIt(indexIntoFileIter_.run(), indexIntoFileIter_.peekAheadAtLumi(), 0U)) {
            return false;
          }
        } while (indexIntoFileIter_.skipLumiInRun());
        return true;
      }
    }
    return false;
  }

  bool RootFile::isDuplicateEvent() {
    assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kEvent);
    if (duplicateChecker_.get() == nullptr) {
      return false;
    }
    auto const eventAux = fillEventAuxiliary(indexIntoFileIter_.entry());
    return duplicateChecker_->isDuplicateAndCheckActive(indexIntoFileIter_.processHistoryIDIndex(),
                                                        indexIntoFileIter_.run(),
                                                        indexIntoFileIter_.lumi(),
                                                        eventAux.id().event(),
                                                        file_);
  }

  bool RootFile::containsItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {
    return indexIntoFile_.containsItem(run, lumi, event);
  }

  IndexIntoFile::EntryType RootFile::getNextItemType(RunNumber_t& run,
                                                     LuminosityBlockNumber_t& lumi,
                                                     EventNumber_t& event) {
    // First, account for consecutive skipped entries.
    while (skipThisEntry()) {
      if (indexIntoFileIter_.getEntryType() == IndexIntoFile::kRun) {
        indexIntoFileIter_.advanceToNextRun();
      } else if (indexIntoFileIter_.getEntryType() == IndexIntoFile::kLumi) {
        indexIntoFileIter_.advanceToNextLumiOrRun();
      } else {
        ++indexIntoFileIter_;
      }
    }
    // OK, we have an entry that is not skipped.
    IndexIntoFile::EntryType entryType = indexIntoFileIter_.getEntryType();
    if (entryType == IndexIntoFile::kEnd) {
      return IndexIntoFile::kEnd;
    }
    if (entryType == IndexIntoFile::kRun) {
      run = indexIntoFileIter_.run();
      runHelper_->checkForNewRun(run, indexIntoFileIter_.peekAheadAtLumi());
      return IndexIntoFile::kRun;
    } else if (processingMode_ == InputSource::Runs) {
      indexIntoFileIter_.advanceToNextRun();
      return getNextItemType(run, lumi, event);
    }
    if (entryType == IndexIntoFile::kLumi) {
      run = indexIntoFileIter_.run();
      lumi = indexIntoFileIter_.lumi();
      return IndexIntoFile::kLumi;
    } else if (processingMode_ == InputSource::RunsAndLumis) {
      indexIntoFileIter_.advanceToNextLumiOrRun();
      return getNextItemType(run, lumi, event);
    }
    if (isDuplicateEvent()) {
      ++indexIntoFileIter_;
      return getNextItemType(run, lumi, event);
    }
    run = indexIntoFileIter_.run();
    lumi = indexIntoFileIter_.lumi();
    auto eventAux = fillEventAuxiliary(indexIntoFileIter_.entry());
    event = eventAux.event();
    return IndexIntoFile::kEvent;
  }

  bool RootFile::wasLastEventJustRead() const {
    IndexIntoFile::IndexIntoFileItr itr(indexIntoFileIter_);
    itr.advanceToEvent();
    return itr.getEntryType() == IndexIntoFile::kEnd;
  }

  bool RootFile::wasFirstEventJustRead() const {
    IndexIntoFile::IndexIntoFileItr itr(indexIntoFileIter_);
    int phIndex;
    RunNumber_t run;
    LuminosityBlockNumber_t lumi;
    IndexIntoFile::EntryNumber_t eventEntry;
    itr.skipEventBackward(phIndex, run, lumi, eventEntry);
    itr.skipEventBackward(phIndex, run, lumi, eventEntry);
    return eventEntry == IndexIntoFile::invalidEntry;
  }

  namespace {
    struct RunItem {
      RunItem(ProcessHistoryID const& phid, RunNumber_t const& run) : phid_(phid), run_(run) {}
      ProcessHistoryID phid_;
      RunNumber_t run_;
    };
    struct RunItemSortByRun {
      bool operator()(RunItem const& a, RunItem const& b) const { return a.run_ < b.run_; }
    };
    struct RunItemSortByRunPhid {
      bool operator()(RunItem const& a, RunItem const& b) const {
        return a.run_ < b.run_ || (!(b.run_ < a.run_) && a.phid_ < b.phid_);
      }
    };
    struct LumiItem {
      LumiItem(ProcessHistoryID const& phid,
               RunNumber_t const& run,
               LuminosityBlockNumber_t const& lumi,
               IndexIntoFile::EntryNumber_t const& entry)
          : phid_(phid),
            run_(run),
            lumi_(lumi),
            firstEventEntry_(entry),
            lastEventEntry_(entry == IndexIntoFile::invalidEntry ? IndexIntoFile::invalidEntry : entry + 1) {}
      ProcessHistoryID phid_;
      RunNumber_t run_;
      LuminosityBlockNumber_t lumi_;
      IndexIntoFile::EntryNumber_t firstEventEntry_;
      IndexIntoFile::EntryNumber_t lastEventEntry_;
    };
    struct LumiItemSortByRunLumi {
      bool operator()(LumiItem const& a, LumiItem const& b) const {
        return a.run_ < b.run_ || (!(b.run_ < a.run_) && a.lumi_ < b.lumi_);
      }
    };
    struct LumiItemSortByRunLumiPhid {
      bool operator()(LumiItem const& a, LumiItem const& b) const {
        if (a.run_ < b.run_)
          return true;
        if (b.run_ < a.run_)
          return false;
        if (a.lumi_ < b.lumi_)
          return true;
        if (b.lumi_ < a.lumi_)
          return false;
        return a.phid_ < b.phid_;
      }
    };
  }  // namespace

  void RootFile::fillIndexIntoFile() {
    // This function is for backward compatibility.
    // If reading a current format file, indexIntoFile_ is read from the input
    // file and should always be there. Note that the algorithm below will work
    // sometimes but often fail with the new format introduced in release 3_8_0.
    // If it ever becomes necessary to rebuild IndexIntoFile from the new format,
    // probably a separate function should be written to deal with the task.
    // This is possible just not implemented yet.
    assert(!fileFormatVersion().hasIndexIntoFile());

    typedef std::list<LumiItem> LumiList;
    LumiList lumis;  // (declare 1)

    typedef std::set<LuminosityBlockID> RunLumiSet;
    RunLumiSet runLumiSet;  // (declare 2)

    typedef std::list<RunItem> RunList;
    RunList runs;  // (declare 5)

    typedef std::set<RunNumber_t> RunSet;
    RunSet runSet;  // (declare 4)

    typedef std::set<RunItem, RunItemSortByRunPhid> RunItemSet;
    RunItemSet runItemSet;  // (declare 3)

    typedef std::map<RunNumber_t, ProcessHistoryID> PHIDMap;
    PHIDMap phidMap;

    RunNumber_t prevRun = 0;
    LuminosityBlockNumber_t prevLumi = 0;
    ProcessHistoryID prevPhid;
    bool iFirst = true;

    indexIntoFile_.unsortedEventNumbers().clear();  // should already be empty, just being careful
    indexIntoFile_.unsortedEventNumbers().reserve(eventTree_.entries());

    // First, loop through the event tree.
    EventSelectionIDVector eventSelectionIDs;
    BranchListIndexes branchListIndexes;
    while (eventTree_.next()) {
      bool newRun = false;
      bool newLumi = false;
      auto evtAux = fillThisEventAuxiliary();
      fillEventHistory(evtAux, eventSelectionIDs, branchListIndexes);

      // Save the event numbers as we loop through the event auxiliary to avoid
      // having to read through the event auxiliary again later. These event numbers
      // are not actually used in this function, but could be needed elsewhere.
      indexIntoFile_.unsortedEventNumbers().push_back(evtAux.event());

      ProcessHistoryID reducedPHID = processHistoryRegistry_->reducedProcessHistoryID(evtAux.processHistoryID());

      if (iFirst || prevPhid != reducedPHID || prevRun != evtAux.run()) {
        iFirst = false;
        newRun = newLumi = true;
      } else if (prevLumi != evtAux.luminosityBlock()) {
        newLumi = true;
      }
      prevPhid = reducedPHID;
      prevRun = evtAux.run();
      prevLumi = evtAux.luminosityBlock();
      if (newLumi) {
        lumis.emplace_back(
            reducedPHID, evtAux.run(), evtAux.luminosityBlock(), eventTree_.entryNumber());  // (insert 1)
        runLumiSet.insert(LuminosityBlockID(evtAux.run(), evtAux.luminosityBlock()));        // (insert 2)
      } else {
        LumiItem& currentLumi = lumis.back();
        assert(currentLumi.lastEventEntry_ == eventTree_.entryNumber());
        ++currentLumi.lastEventEntry_;
      }
      if (newRun) {
        // Insert run in list if it is not already there.
        RunItem item(reducedPHID, evtAux.run());
        if (runItemSet.insert(item).second) {  // (check 3, insert 3)
          runs.push_back(std::move(item));     // (insert 5)
          runSet.insert(evtAux.run());         // (insert 4)
          phidMap.insert(std::make_pair(evtAux.run(), reducedPHID));
        }
      }
    }
    // now clean up.
    eventTree_.setEntryNumber(IndexIntoFile::invalidEntry);
    lastEventEntryNumberRead_ = IndexIntoFile::invalidEntry;

    // Loop over run entries and fill information.

    typedef std::map<RunNumber_t, IndexIntoFile::EntryNumber_t> RunMap;
    RunMap runMap;  // (declare 11)

    typedef std::vector<RunItem> RunVector;
    RunVector emptyRuns;  // (declare 12)

    if (runTree_.isValid()) {
      while (runTree_.next()) {
        // Note: adjacent duplicates will be skipped without an explicit check.

        std::shared_ptr<RunAuxiliary> runAux = fillRunAuxiliary();
        ProcessHistoryID reducedPHID = processHistoryRegistry_->reducedProcessHistoryID(runAux->processHistoryID());

        if (runSet.insert(runAux->run()).second) {  // (check 4, insert 4)
          // This run was not associated with any events.
          emptyRuns.emplace_back(reducedPHID, runAux->run());  // (insert 12)
        }
        runMap.insert(std::make_pair(runAux->run(), runTree_.entryNumber()));  // (insert 11)
        phidMap.insert(std::make_pair(runAux->run(), reducedPHID));
      }
      // now clean up.
      runTree_.setEntryNumber(IndexIntoFile::invalidEntry);
    }

    // Insert the ordered empty runs into the run list.
    RunItemSortByRun runItemSortByRun;
    stable_sort_all(emptyRuns, runItemSortByRun);

    RunList::iterator itRuns = runs.begin(), endRuns = runs.end();
    for (auto const& emptyRun : emptyRuns) {
      for (; itRuns != endRuns; ++itRuns) {
        if (runItemSortByRun(emptyRun, *itRuns)) {
          break;
        }
      }
      runs.insert(itRuns, emptyRun);
    }

    // Loop over luminosity block entries and fill information.

    typedef std::vector<LumiItem> LumiVector;
    LumiVector emptyLumis;  // (declare 7)

    typedef std::map<LuminosityBlockID, IndexIntoFile::EntryNumber_t> RunLumiMap;
    RunLumiMap runLumiMap;  // (declare 6)

    if (lumiTree_.isValid()) {
      while (lumiTree_.next()) {
        // Note: adjacent duplicates will be skipped without an explicit check.
        std::shared_ptr<LuminosityBlockAuxiliary> lumiAux = fillLumiAuxiliary();
        LuminosityBlockID lumiID = LuminosityBlockID(lumiAux->run(), lumiAux->luminosityBlock());
        if (runLumiSet.insert(lumiID).second) {  // (check 2, insert 2)
          // This lumi was not associated with any events.
          // Use the process history ID from the corresponding run.  In cases of practical
          // importance, this should be the correct process history ID,  but it is possible
          // to construct files where this is not the correct process history ID ...
          PHIDMap::const_iterator iPhidMap = phidMap.find(lumiAux->run());
          assert(iPhidMap != phidMap.end());
          emptyLumis.emplace_back(
              iPhidMap->second, lumiAux->run(), lumiAux->luminosityBlock(), IndexIntoFile::invalidEntry);  // (insert 7)
        }
        runLumiMap.insert(std::make_pair(lumiID, lumiTree_.entryNumber()));
      }
      // now clean up.
      lumiTree_.setEntryNumber(IndexIntoFile::invalidEntry);
    }

    // Insert the ordered empty lumis into the lumi list.
    LumiItemSortByRunLumi lumiItemSortByRunLumi;
    stable_sort_all(emptyLumis, lumiItemSortByRunLumi);

    LumiList::iterator itLumis = lumis.begin(), endLumis = lumis.end();
    for (auto const& emptyLumi : emptyLumis) {
      for (; itLumis != endLumis; ++itLumis) {
        if (lumiItemSortByRunLumi(emptyLumi, *itLumis)) {
          break;
        }
      }
      lumis.insert(itLumis, emptyLumi);
    }

    // Create a map of RunItems that gives the order of first appearance in the list.
    // Also fill in the vector of process history IDs
    typedef std::map<RunItem, int, RunItemSortByRunPhid> RunCountMap;
    RunCountMap runCountMap;  // Declare (17)
    std::vector<ProcessHistoryID>& phids = indexIntoFile_.setProcessHistoryIDs();
    assert(phids.empty());
    std::vector<IndexIntoFile::RunOrLumiEntry>& entries = indexIntoFile_.setRunOrLumiEntries();
    assert(entries.empty());
    int rcount = 0;
    for (auto& run : runs) {
      RunCountMap::const_iterator countMapItem = runCountMap.find(run);
      if (countMapItem == runCountMap.end()) {
        countMapItem = runCountMap.insert(std::make_pair(run, rcount)).first;  // Insert (17)
        assert(countMapItem != runCountMap.end());
        ++rcount;
      }
      std::vector<ProcessHistoryID>::const_iterator phidItem = find_in_all(phids, run.phid_);
      if (phidItem == phids.end()) {
        phids.push_back(run.phid_);
        phidItem = phids.end() - 1;
      }
      entries.emplace_back(countMapItem->second,  // use (17)
                           IndexIntoFile::invalidEntry,
                           runMap[run.run_],  // use (11)
                           phidItem - phids.begin(),
                           run.run_,
                           0U,
                           IndexIntoFile::invalidEntry,
                           IndexIntoFile::invalidEntry);
    }

    // Create a map of LumiItems that gives the order of first appearance in the list.
    typedef std::map<LumiItem, int, LumiItemSortByRunLumiPhid> LumiCountMap;
    LumiCountMap lumiCountMap;  // Declare (19)
    int lcount = 0;
    for (auto& lumi : lumis) {
      RunCountMap::const_iterator runCountMapItem = runCountMap.find(RunItem(lumi.phid_, lumi.run_));
      assert(runCountMapItem != runCountMap.end());
      LumiCountMap::const_iterator countMapItem = lumiCountMap.find(lumi);
      if (countMapItem == lumiCountMap.end()) {
        countMapItem = lumiCountMap.insert(std::make_pair(lumi, lcount)).first;  // Insert (17)
        assert(countMapItem != lumiCountMap.end());
        ++lcount;
      }
      std::vector<ProcessHistoryID>::const_iterator phidItem = find_in_all(phids, lumi.phid_);
      assert(phidItem != phids.end());
      entries.emplace_back(runCountMapItem->second,
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

  void RootFile::validateFile(InputType inputType, bool usingGoToEvent) {
    if (!fid_.isValid()) {
      fid_ = FileID(createGlobalIdentifier());
    }
    if (!eventTree_.isValid()) {
      throw Exception(errors::EventCorruption) << "'Events' tree is corrupted or not present\n"
                                               << "in the input file.\n";
    }
    if (enforceGUIDInFileName_) {
      auto guidFromName = stemFromPath(file_);
      if (guidFromName != fid_.fid()) {
        throw edm::Exception(edm::errors::FileNameInconsistentWithGUID)
            << "GUID " << guidFromName << " extracted from file name " << file_
            << " is inconsistent with the GUID read from the file " << fid_.fid();
      }
    }

    if (fileFormatVersion().hasIndexIntoFile()) {
      if (runTree().entries() > 0) {
        assert(!indexIntoFile_.empty());
      }
      if (!fileFormatVersion().useReducedProcessHistoryID()) {
        if (daqProvenanceHelper_) {
          std::vector<ProcessHistoryID>& phidVec = indexIntoFile_.setProcessHistoryIDs();
          for (auto& phid : phidVec) {
            phid = daqProvenanceHelper_->mapProcessHistoryID(phid);
          }
        }
        indexIntoFile_.reduceProcessHistoryIDs(*processHistoryRegistry_);
      }
    } else {
      assert(indexIntoFile_.empty());
      fillIndexIntoFile();
    }

    indexIntoFile_.fixIndexes(orderedProcessHistoryIDs_);
    indexIntoFile_.setNumberOfEvents(eventTree_.entries());
    indexIntoFile_.setEventFinder(
        std::shared_ptr<IndexIntoFile::EventFinder>(std::make_shared<RootFileEventFinder>(eventTree_)));
    // We fill the event numbers explicitly if we need to find events in closed files,
    // such as for secondary files (or secondary sources) or if duplicate checking across files.
    bool needEventNumbers = false;
    bool needIndexesForDuplicateChecker =
        duplicateChecker_ && duplicateChecker_->checkingAllFiles() && !duplicateChecker_->checkDisabled();
    if (inputType != InputType::Primary || needIndexesForDuplicateChecker || usingGoToEvent) {
      needEventNumbers = true;
    }
    bool needEventEntries = false;
    if (inputType != InputType::Primary || !noEventSort_) {
      // We need event entries for sorting or for secondary files or sources.
      needEventEntries = true;
    }
    indexIntoFile_.fillEventNumbersOrEntries(needEventNumbers, needEventEntries);
  }

  void RootFile::reportOpened(std::string const& inputType) {
    // Report file opened.
    std::string const label = "source";
    std::string moduleName = "PoolSource";
    filePtr_->inputFileOpened(logicalFile_, inputType, moduleName, label, fid_.fid(), eventTree_.branchNames());
  }

  void RootFile::close() {
    // Just to play it safe, zero all pointers to objects in the InputFile to be closed.
    eventHistoryTree_ = nullptr;
    for (auto& treePointer : treePointers_) {
      treePointer->close();
      treePointer = nullptr;
    }
    filePtr_->Close();
    filePtr_ = nullptr;  // propagate_const<T> has no reset() function
  }

  EventAuxiliary const& RootFile::fillThisEventAuxiliary() {
    if (lastEventEntryNumberRead_ == eventTree_.entryNumber()) {
      // Already read.
      return eventAuxCache_;
    }
    if (fileFormatVersion().newAuxiliary()) {
      EventAuxiliary* pEvAux = &eventAuxCache_;
      eventTree_.fillAux<EventAuxiliary>(pEvAux);
    } else {
      // for backward compatibility.
      EventAux eventAux;
      EventAux* pEvAux = &eventAux;
      eventTree_.fillAux<EventAux>(pEvAux);
      conversion(eventAux, eventAuxCache_);
    }
    lastEventEntryNumberRead_ = eventTree_.entryNumber();
    return eventAuxCache_;
  }

  EventAuxiliary RootFile::fillEventAuxiliary(IndexIntoFile::EntryNumber_t entry) {
    eventTree_.setEntryNumber(entry);
    return fillThisEventAuxiliary();
  }

  void RootFile::fillEventToProcessBlockIndexes() {
    TBranch* eventToProcessBlockIndexesBranch = get_underlying_safe(eventToProcessBlockIndexesBranch_);
    if (eventToProcessBlockIndexesBranch == nullptr) {
      if (processBlockHelper_.get() == nullptr) {
        eventToProcessBlockIndexes_.setIndex(0);
      } else {
        eventToProcessBlockIndexes_.setIndex(processBlockHelper_->outerOffset());
      }
    } else {
      if (processBlockHelper_->cacheIndexVectorsPerFile().back() == 1u) {
        eventToProcessBlockIndexes_.setIndex(processBlockHelper_->outerOffset());
      } else {
        EventToProcessBlockIndexes* pEventToProcessBlockIndexes = &eventToProcessBlockIndexes_;
        eventTree_.fillBranchEntry(eventToProcessBlockIndexesBranch, pEventToProcessBlockIndexes);
        unsigned int updatedIndex = eventToProcessBlockIndexes_.index() + processBlockHelper_->outerOffset();
        eventToProcessBlockIndexes_.setIndex(updatedIndex);
      }
    }
  }

  bool RootFile::fillEventHistory(EventAuxiliary& evtAux,
                                  EventSelectionIDVector& eventSelectionIDs,
                                  BranchListIndexes& branchListIndexes,
                                  bool assertOnFailure) {
    // We could consider doing delayed reading, but because we have to
    // store this History object in a different tree than the event
    // data tree, this is too hard to do in this first version.
    if (fileFormatVersion().eventHistoryBranch()) {
      // Lumi block number was not in EventID for the relevant releases.
      EventID id(evtAux.id().run(), 0, evtAux.id().event());
      if (eventProcessHistoryIter_->eventID() != id) {
        EventProcessHistoryID target(id, ProcessHistoryID());
        eventProcessHistoryIter_ = lower_bound_all(eventProcessHistoryIDs_, target);
        assert(eventProcessHistoryIter_->eventID() == id);
      }
      evtAux.setProcessHistoryID(eventProcessHistoryIter_->processHistoryID());
      ++eventProcessHistoryIter_;
    } else if (fileFormatVersion().eventHistoryTree()) {
      // for backward compatibility.
      History* pHistory = history_.get();
      TBranch* eventHistoryBranch = eventHistoryTree_->GetBranch(poolNames::eventHistoryBranchName().c_str());
      if (!eventHistoryBranch) {
        throw Exception(errors::EventCorruption) << "Failed to find history branch in event history tree.\n";
      }
      eventHistoryBranch->SetAddress(&pHistory);
      roottree::getEntry(eventHistoryTree_, eventTree_.entryNumber());
      evtAux.setProcessHistoryID(history_->processHistoryID());
      eventSelectionIDs.swap(history_->eventSelectionIDs());
      branchListIndexes.swap(history_->branchListIndexes());
    } else if (fileFormatVersion().noMetaDataTrees()) {
      // Current format
      EventSelectionIDVector* pESV = &eventSelectionIDs;
      TBranch* eventSelectionIDBranch = eventTree_.tree()->GetBranch(poolNames::eventSelectionsBranchName().c_str());
      assert(eventSelectionIDBranch != nullptr);
      eventTree_.fillBranchEntry(eventSelectionIDBranch, pESV);
      BranchListIndexes* pBLI = &branchListIndexes;
      TBranch* branchListIndexesBranch = eventTree_.tree()->GetBranch(poolNames::branchListIndexesBranchName().c_str());
      assert(branchListIndexesBranch != nullptr);
      eventTree_.fillBranchEntry(branchListIndexesBranch, pBLI);
    }
    if (provenanceAdaptor_) {
      evtAux.setProcessHistoryID(provenanceAdaptor_->convertID(evtAux.processHistoryID()));
      for (auto& esID : eventSelectionIDs) {
        esID = provenanceAdaptor_->convertID(esID);
      }
    }
    if (daqProvenanceHelper_) {
      evtAux.setProcessHistoryID(daqProvenanceHelper_->mapProcessHistoryID(evtAux.processHistoryID()));
    }
    if (!fileFormatVersion().splitProductIDs()) {
      // old format.  branchListIndexes_ must be filled in from the ProvenanceAdaptor.
      provenanceAdaptor_->branchListIndexes(branchListIndexes);
    }
    if (branchIDListHelper_) {
      return branchIDListHelper_->fixBranchListIndexes(branchListIndexes, assertOnFailure);
    }
    return true;
  }

  std::shared_ptr<LuminosityBlockAuxiliary> RootFile::fillLumiAuxiliary() {
    auto lumiAuxiliary = std::make_shared<LuminosityBlockAuxiliary>();
    if (fileFormatVersion().newAuxiliary()) {
      LuminosityBlockAuxiliary* pLumiAux = lumiAuxiliary.get();
      lumiTree_.fillAux<LuminosityBlockAuxiliary>(pLumiAux);
    } else {
      LuminosityBlockAux lumiAux;
      LuminosityBlockAux* pLumiAux = &lumiAux;
      lumiTree_.fillAux<LuminosityBlockAux>(pLumiAux);
      conversion(lumiAux, *lumiAuxiliary);
    }
    if (provenanceAdaptor_) {
      lumiAuxiliary->setProcessHistoryID(provenanceAdaptor_->convertID(lumiAuxiliary->processHistoryID()));
    }
    if (daqProvenanceHelper_) {
      lumiAuxiliary->setProcessHistoryID(daqProvenanceHelper_->mapProcessHistoryID(lumiAuxiliary->processHistoryID()));
    }
    if (lumiAuxiliary->luminosityBlock() == 0 && !fileFormatVersion().runsAndLumis()) {
      lumiAuxiliary->id() = LuminosityBlockID(RunNumber_t(1), LuminosityBlockNumber_t(1));
    }
    return lumiAuxiliary;
  }

  std::shared_ptr<RunAuxiliary> RootFile::fillRunAuxiliary() {
    auto runAuxiliary = std::make_shared<RunAuxiliary>();
    if (fileFormatVersion().newAuxiliary()) {
      RunAuxiliary* pRunAux = runAuxiliary.get();
      runTree_.fillAux<RunAuxiliary>(pRunAux);
    } else {
      RunAux runAux;
      RunAux* pRunAux = &runAux;
      runTree_.fillAux<RunAux>(pRunAux);
      conversion(runAux, *runAuxiliary);
    }
    if (provenanceAdaptor_) {
      runAuxiliary->setProcessHistoryID(provenanceAdaptor_->convertID(runAuxiliary->processHistoryID()));
    }
    if (daqProvenanceHelper_) {
      runAuxiliary->setProcessHistoryID(daqProvenanceHelper_->mapProcessHistoryID(runAuxiliary->processHistoryID()));
    }
    return runAuxiliary;
  }

  bool RootFile::skipEvents(int& offset) {
    while (offset > 0 && indexIntoFileIter_ != indexIntoFileEnd_) {
      int phIndexOfSkippedEvent = IndexIntoFile::invalidIndex;
      RunNumber_t runOfSkippedEvent = IndexIntoFile::invalidRun;
      LuminosityBlockNumber_t lumiOfSkippedEvent = IndexIntoFile::invalidLumi;
      IndexIntoFile::EntryNumber_t skippedEventEntry = IndexIntoFile::invalidEntry;

      indexIntoFileIter_.skipEventForward(
          phIndexOfSkippedEvent, runOfSkippedEvent, lumiOfSkippedEvent, skippedEventEntry);

      // At the end of the file and there were no more events to skip
      if (skippedEventEntry == IndexIntoFile::invalidEntry)
        break;

      if (eventSkipperByID_ && eventSkipperByID_->somethingToSkip()) {
        auto const evtAux = fillEventAuxiliary(skippedEventEntry);
        if (eventSkipperByID_->skipIt(runOfSkippedEvent, lumiOfSkippedEvent, evtAux.id().event())) {
          continue;
        }
      }
      if (duplicateChecker_ && !duplicateChecker_->checkDisabled() && !duplicateChecker_->noDuplicatesInFile()) {
        auto const evtAux = fillEventAuxiliary(skippedEventEntry);
        if (duplicateChecker_->isDuplicateAndCheckActive(
                phIndexOfSkippedEvent, runOfSkippedEvent, lumiOfSkippedEvent, evtAux.id().event(), file_)) {
          continue;
        }
      }
      --offset;
    }

    while (offset < 0) {
      if (duplicateChecker_) {
        duplicateChecker_->disable();
      }

      int phIndexOfEvent = IndexIntoFile::invalidIndex;
      RunNumber_t runOfEvent = IndexIntoFile::invalidRun;
      LuminosityBlockNumber_t lumiOfEvent = IndexIntoFile::invalidLumi;
      IndexIntoFile::EntryNumber_t eventEntry = IndexIntoFile::invalidEntry;

      indexIntoFileIter_.skipEventBackward(phIndexOfEvent, runOfEvent, lumiOfEvent, eventEntry);

      if (eventEntry == IndexIntoFile::invalidEntry)
        break;

      if (eventSkipperByID_ && eventSkipperByID_->somethingToSkip()) {
        auto const evtAux = fillEventAuxiliary(eventEntry);
        if (eventSkipperByID_->skipIt(runOfEvent, lumiOfEvent, evtAux.id().event())) {
          continue;
        }
      }
      ++offset;
    }
    return (indexIntoFileIter_ == indexIntoFileEnd_);
  }

  bool RootFile::goToEvent(EventID const& eventID) {
    indexIntoFile_.fillEventNumbers();

    if (duplicateChecker_) {
      duplicateChecker_->disable();
    }

    IndexIntoFile::SortOrder sortOrder = IndexIntoFile::numericalOrder;
    if (noEventSort_)
      sortOrder = IndexIntoFile::firstAppearanceOrder;
    if (noRunLumiSort_) {
      sortOrder = IndexIntoFile::entryOrder;
    }

    IndexIntoFile::IndexIntoFileItr iter =
        indexIntoFile_.findPosition(sortOrder, eventID.run(), eventID.luminosityBlock(), eventID.event());

    if (iter == indexIntoFile_.end(sortOrder)) {
      return false;
    }
    indexIntoFileIter_ = iter;
    return true;
  }

  // readEvent() is responsible for creating, and setting up, the
  // EventPrincipal.
  //
  //   1. create an EventPrincipal with a unique EventID
  //   2. For each entry in the provenance, put in one ProductResolver,
  //      holding the Provenance for the corresponding EDProduct.
  //   3. set up the the EventPrincipal to know about this ProductResolver.
  //
  // We do *not* create the EDProduct instance (the equivalent of reading
  // the branch containing this EDProduct. That will be done by the Delayed Reader,
  //  when it is asked to do so.
  //
  bool RootFile::readEvent(EventPrincipal& principal) {
    assert(indexIntoFileIter_ != indexIntoFileEnd_);
    assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kEvent);
    // read the event
    auto [found, succeeded] = readCurrentEvent(principal, false);
    auto const& evtAux = principal.aux();

    runHelper_->checkRunConsistency(evtAux.run(), indexIntoFileIter_.run());
    runHelper_->checkLumiConsistency(evtAux.luminosityBlock(), indexIntoFileIter_.lumi());

    ++indexIntoFileIter_;
    return succeeded;
  }

  // Reads event at the current entry in the event tree
  std::tuple<bool, bool> RootFile::readCurrentEvent(EventPrincipal& principal, bool assertOnFailure) {
    bool found = true;
    bool succeeded = true;
    if (!eventTree_.current()) {
      found = false;
      return {found, succeeded};
    }
    auto evtAux = fillThisEventAuxiliary();
    if (!fileFormatVersion().lumiInEventID()) {
      //ugly, but will disappear when the backward compatibility is done with schema evolution.
      const_cast<EventID&>(evtAux.id()).setLuminosityBlockNumber(evtAux.oldLuminosityBlock());
      evtAux.resetObsoleteInfo();
    }
    fillEventToProcessBlockIndexes();
    EventSelectionIDVector eventSelectionIDs;
    BranchListIndexes branchListIndexes;
    if (!fillEventHistory(evtAux, eventSelectionIDs, branchListIndexes, assertOnFailure)) {
      succeeded = false;
    }
    runHelper_->overrideRunNumber(evtAux.id(), evtAux.isRealData());

    // We're not done ... so prepare the EventPrincipal
    eventTree_.insertEntryForIndex(principal.transitionIndex());
    auto history = processHistoryRegistry_->getMapped(evtAux.processHistoryID());
    principal.fillEventPrincipal(evtAux,
                                 history,
                                 std::move(eventSelectionIDs),
                                 std::move(branchListIndexes),
                                 eventToProcessBlockIndexes_,
                                 *(makeProductProvenanceRetriever(principal.streamID().value())),
                                 eventTree_.resetAndGetRootDelayedReader());

    // If this next assert shows up in performance profiling or significantly affects memory, then these three lines should be deleted.
    // The IndexIntoFile should guarantee that it never fails.
    ProcessHistoryID idToCheck = (daqProvenanceHelper_ && fileFormatVersion().useReducedProcessHistoryID()
                                      ? *daqProvenanceHelper_->oldProcessHistoryID()
                                      : evtAux.processHistoryID());
    ProcessHistoryID const& reducedPHID = processHistoryRegistry_->reducedProcessHistoryID(idToCheck);
    assert(reducedPHID == indexIntoFile_.processHistoryID(indexIntoFileIter_.processHistoryIDIndex()));

    // report event read from file
    filePtr_->eventReadFromFile();
    return {found, succeeded};
  }

  void RootFile::setAtEventEntry(IndexIntoFile::EntryNumber_t entry) { eventTree_.setEntryNumber(entry); }

  std::shared_ptr<RunAuxiliary> RootFile::readRunAuxiliary_() {
    if (runHelper_->fakeNewRun()) {
      auto runAuxiliary = std::make_shared<RunAuxiliary>(*savedRunAuxiliary());
      runHelper_->overrideRunNumber(runAuxiliary->id());
      return runAuxiliary;
    }
    assert(indexIntoFileIter_ != indexIntoFileEnd_);
    assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kRun);

    // Begin code for backward compatibility before the existence of run trees.
    if (!runTree_.isValid()) {
      // prior to the support of run trees.
      // RunAuxiliary did not contain a valid timestamp.  Take it from the next event.
      IndexIntoFile::EntryNumber_t eventEntry = indexIntoFileIter_.firstEventEntryThisRun();
      assert(eventEntry != IndexIntoFile::invalidEntry);
      assert(eventTree_.current(eventEntry));
      auto const evtAux = fillEventAuxiliary(eventEntry);

      RunID run = RunID(indexIntoFileIter_.run());
      runHelper_->overrideRunNumber(run);
      savedRunAuxiliary_ = std::make_shared<RunAuxiliary>(run.run(), evtAux.time(), Timestamp::invalidTimestamp());
      return savedRunAuxiliary();
    }
    // End code for backward compatibility before the existence of run trees.
    runTree_.setEntryNumber(indexIntoFileIter_.entry());
    std::shared_ptr<RunAuxiliary> runAuxiliary = fillRunAuxiliary();
    assert(runAuxiliary->run() == indexIntoFileIter_.run());
    runHelper_->overrideRunNumber(runAuxiliary->id());
    filePtr_->reportInputRunNumber(runAuxiliary->run());
    // If RunAuxiliary did not contain a valid begin timestamp, invalidate any end timestamp.
    if (runAuxiliary->beginTime() == Timestamp::invalidTimestamp()) {
      runAuxiliary->setEndTime(Timestamp::invalidTimestamp());
    }

    // If RunAuxiliary did not contain a valid timestamp, or if this an old format file from
    // when the Run's ProcessHistory included only processes where products were added to the Run itself,
    // we attempt to read the first event in the run to get appropriate info.
    if (runAuxiliary->beginTime() == Timestamp::invalidTimestamp() ||
        !fileFormatVersion().processHistorySameWithinRun()) {
      IndexIntoFile::EntryNumber_t eventEntry = indexIntoFileIter_.firstEventEntryThisRun();
      // If we have a valid event, use its information.
      if (eventEntry != IndexIntoFile::invalidEntry) {
        assert(eventTree_.current(eventEntry));
        auto evtAux = fillEventAuxiliary(eventEntry);

        // RunAuxiliary did not contain a valid timestamp.  Take it from the next event in this run if there is one.
        if (runAuxiliary->beginTime() == Timestamp::invalidTimestamp()) {
          runAuxiliary->setBeginTime(evtAux.time());
        }

        // For backwards compatibility when the Run's ProcessHistory included only processes where products were added to the
        // Run, and then the Run and Event auxiliaries could be different.  Use the event ProcessHistoryID if there is one. It should
        // almost always be correct by the current definition (processes included if any products are added. This makes the run, lumi,
        // and event ProcessHistory's always be the same if no file merging occurs).
        if (!fileFormatVersion().processHistorySameWithinRun()) {
          EventSelectionIDVector eventSelectionIDs;
          BranchListIndexes branchListIndexes;
          fillEventHistory(evtAux, eventSelectionIDs, branchListIndexes);
          runAuxiliary->setProcessHistoryID(evtAux.processHistoryID());
        }
      }
    }
    savedRunAuxiliary_ = runAuxiliary;
    return runAuxiliary;
  }

  void RootFile::fillProcessBlockHelper_() {
    assert(inputType_ == InputType::Primary);
    std::vector<unsigned int> nEntries;
    nEntries.reserve(processBlockTrees_.size());
    for (auto const& processBlockTree : processBlockTrees_) {
      nEntries.push_back(processBlockTree->entries());
    }
    processBlockHelper_->fillFromPrimaryInput(*storedProcessBlockHelper_, nEntries);
    storedProcessBlockHelper_ =
        std::make_unique<StoredProcessBlockHelper>();  // propagate_const<T> has no reset() function
  }

  bool RootFile::initializeFirstProcessBlockEntry() {
    if (processBlockTrees_[currentProcessBlockTree_]->entryNumber() == IndexIntoFile::invalidEntry) {
      processBlockTrees_[currentProcessBlockTree_]->setEntryNumber(0);
      assert(processBlockTrees_[currentProcessBlockTree_]->current());
      return true;
    }
    return false;
  }

  bool RootFile::endOfProcessBlocksReached() const { return currentProcessBlockTree_ >= processBlockTrees_.size(); }

  bool RootFile::nextProcessBlock_(ProcessBlockPrincipal&) {
    assert(inputType_ == InputType::Primary);
    if (endOfProcessBlocksReached()) {
      return false;
    }
    if (initializeFirstProcessBlockEntry()) {
      return true;
    }
    // With the current design, the RootFile should always be
    // set to a valid ProcessBlock entry in one of the TTrees
    // if it not at the end.
    assert(processBlockTrees_[currentProcessBlockTree_]->current());
    // Try for next entry in the same TTree
    if (processBlockTrees_[currentProcessBlockTree_]->nextWithCache()) {
      return true;
    }
    // Next ProcessBlock TTree
    ++currentProcessBlockTree_;
    if (endOfProcessBlocksReached()) {
      return false;
    }
    // With current design there should always be at least one entry.
    // Initialize for that entry.
    processBlockTrees_[currentProcessBlockTree_]->setEntryNumber(0);
    assert(processBlockTrees_[currentProcessBlockTree_]->current());
    return true;
  }

  void RootFile::readProcessBlock_(ProcessBlockPrincipal& processBlockPrincipal) {
    assert(inputType_ == InputType::Primary);
    RootTree* rootTree = processBlockTrees_[currentProcessBlockTree_].get();
    rootTree->insertEntryForIndex(0);
    assert(!rootTree->processName().empty());
    processBlockPrincipal.fillProcessBlockPrincipal(rootTree->processName(), rootTree->resetAndGetRootDelayedReader());
  }

  bool RootFile::readRun_(RunPrincipal& runPrincipal) {
    bool shouldProcessRun = indexIntoFileIter_.shouldProcessRun();

    MergeableRunProductMetadata* mergeableRunProductMetadata = nullptr;
    if (shouldProcessRun) {
      if (inputType_ == InputType::Primary) {
        mergeableRunProductMetadata = runPrincipal.mergeableRunProductMetadata();
        RootTree::EntryNumber const& entryNumber = runTree_.entryNumber();
        assert(entryNumber >= 0);
        mergeableRunProductMetadata->readRun(
            entryNumber, *storedMergeableRunProductMetadata_, IndexIntoFileItrHolder(indexIntoFileIter_));
      }
    }

    if (!runHelper_->fakeNewRun()) {
      assert(indexIntoFileIter_ != indexIntoFileEnd_);
      assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kRun);
      ++indexIntoFileIter_;
    }
    // Begin code for backward compatibility before the existence of run trees.
    if (!runTree_.isValid()) {
      return shouldProcessRun;
    }
    // End code for backward compatibility before the existence of run trees.
    if (shouldProcessRun) {
      // NOTE: we use 0 for the index since do not do delayed reads for RunPrincipals
      runTree_.insertEntryForIndex(0);
      runPrincipal.fillRunPrincipal(*processHistoryRegistry_, runTree_.resetAndGetRootDelayedReader());
      // Read in all the products now.
      runPrincipal.readAllFromSourceAndMergeImmediately(mergeableRunProductMetadata);
      runPrincipal.setShouldWriteRun(RunPrincipal::kYes);
    } else {
      runPrincipal.fillRunPrincipal(*processHistoryRegistry_, nullptr);
      if (runPrincipal.shouldWriteRun() != RunPrincipal::kYes) {
        runPrincipal.setShouldWriteRun(RunPrincipal::kNo);
      }
    }
    return shouldProcessRun;
  }

  std::shared_ptr<LuminosityBlockAuxiliary> RootFile::readLuminosityBlockAuxiliary_() {
    assert(indexIntoFileIter_ != indexIntoFileEnd_);
    assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kLumi);
    // Begin code for backward compatibility before the existence of lumi trees.
    if (!lumiTree_.isValid()) {
      IndexIntoFile::EntryNumber_t eventEntry = indexIntoFileIter_.firstEventEntryThisLumi();
      assert(eventEntry != IndexIntoFile::invalidEntry);
      assert(eventTree_.current(eventEntry));
      auto const evtAux = fillEventAuxiliary(eventEntry);

      LuminosityBlockID lumi = LuminosityBlockID(indexIntoFileIter_.run(), indexIntoFileIter_.lumi());
      runHelper_->overrideRunNumber(lumi);
      return std::make_shared<LuminosityBlockAuxiliary>(
          lumi.run(), lumi.luminosityBlock(), evtAux.time(), Timestamp::invalidTimestamp());
    }
    // End code for backward compatibility before the existence of lumi trees.
    lumiTree_.setEntryNumber(indexIntoFileIter_.entry());
    std::shared_ptr<LuminosityBlockAuxiliary> lumiAuxiliary = fillLumiAuxiliary();
    assert(lumiAuxiliary->run() == indexIntoFileIter_.run());
    assert(lumiAuxiliary->luminosityBlock() == indexIntoFileIter_.lumi());
    runHelper_->overrideRunNumber(lumiAuxiliary->id());
    filePtr_->reportInputLumiSection(lumiAuxiliary->run(), lumiAuxiliary->luminosityBlock());
    if (lumiAuxiliary->beginTime() == Timestamp::invalidTimestamp()) {
      IndexIntoFile::EntryNumber_t eventEntry = indexIntoFileIter_.firstEventEntryThisLumi();
      if (eventEntry != IndexIntoFile::invalidEntry) {
        assert(eventTree_.current(eventEntry));
        auto const evtAux = fillEventAuxiliary(eventEntry);

        lumiAuxiliary->setBeginTime(evtAux.time());
      }
      lumiAuxiliary->setEndTime(Timestamp::invalidTimestamp());
    }
    if (!fileFormatVersion().processHistorySameWithinRun() && savedRunAuxiliary_) {
      lumiAuxiliary->setProcessHistoryID(savedRunAuxiliary_->processHistoryID());
    }
    return lumiAuxiliary;
  }

  bool RootFile::readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal) {
    bool shouldProcessLumi = indexIntoFileIter_.shouldProcessLumi();
    assert(indexIntoFileIter_ != indexIntoFileEnd_);
    assert(indexIntoFileIter_.getEntryType() == IndexIntoFile::kLumi);
    // Begin code for backward compatibility before the existence of lumi trees.
    if (!lumiTree_.isValid()) {
      ++indexIntoFileIter_;
      return shouldProcessLumi;
    }
    // End code for backward compatibility before the existence of lumi trees.
    if (shouldProcessLumi) {
      lumiTree_.setEntryNumber(indexIntoFileIter_.entry());
      // NOTE: we use 0 for the index since do not do delayed reads for LuminosityBlockPrincipals
      lumiTree_.insertEntryForIndex(0);
      auto history = processHistoryRegistry_->getMapped(lumiPrincipal.aux().processHistoryID());
      lumiPrincipal.fillLuminosityBlockPrincipal(history, lumiTree_.resetAndGetRootDelayedReader());
      // Read in all the products now.
      lumiPrincipal.readAllFromSourceAndMergeImmediately();
      lumiPrincipal.setShouldWriteLumi(LuminosityBlockPrincipal::kYes);
    } else {
      auto history = processHistoryRegistry_->getMapped(lumiPrincipal.aux().processHistoryID());
      lumiPrincipal.fillLuminosityBlockPrincipal(history, nullptr);
      if (lumiPrincipal.shouldWriteLumi() != LuminosityBlockPrincipal::kYes) {
        lumiPrincipal.setShouldWriteLumi(LuminosityBlockPrincipal::kNo);
      }
    }
    ++indexIntoFileIter_;
    return shouldProcessLumi;
  }

  bool RootFile::setEntryAtEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) {
    indexIntoFileIter_ = indexIntoFile_.findEventPosition(run, lumi, event);
    if (indexIntoFileIter_ == indexIntoFileEnd_)
      return false;
    eventTree_.setEntryNumber(indexIntoFileIter_.entry());
    return true;
  }

  bool RootFile::setEntryAtLumi(RunNumber_t run, LuminosityBlockNumber_t lumi) {
    indexIntoFileIter_ = indexIntoFile_.findLumiPosition(run, lumi);
    if (indexIntoFileIter_ == indexIntoFileEnd_)
      return false;
    lumiTree_.setEntryNumber(indexIntoFileIter_.entry());
    return true;
  }

  bool RootFile::setEntryAtRun(RunNumber_t run) {
    indexIntoFileIter_ = indexIntoFile_.findRunPosition(run);
    if (indexIntoFileIter_ == indexIntoFileEnd_)
      return false;
    runTree_.setEntryNumber(indexIntoFileIter_.entry());
    return true;
  }

  bool RootFile::setEntryAtNextEventInLumi(RunNumber_t run, LuminosityBlockNumber_t lumi) {
    if (indexIntoFileIter_.getEntryType() == IndexIntoFile::kEvent) {
      ++indexIntoFileIter_;
    }
    indexIntoFileIter_.advanceToEvent();
    if (indexIntoFileIter_.getEntryType() != IndexIntoFile::kEvent)
      return false;
    if (run != indexIntoFileIter_.run())
      return false;
    if (lumi != indexIntoFileIter_.lumi())
      return false;
    //The following is used for its side effect of advancing the
    // eventTree entry.
    fillEventAuxiliary(indexIntoFileIter_.entry());
    return true;
  }

  void RootFile::readEventHistoryTree() {
    // Read in the event history tree, if we have one...
    if (fileFormatVersion().eventHistoryTree()) {
      history_ = std::make_unique<History>();  // propagate_const<T> has no reset() function
      eventHistoryTree_ = dynamic_cast<TTree*>(filePtr_->Get(poolNames::eventHistoryTreeName().c_str()));
      if (!eventHistoryTree_) {
        throw Exception(errors::EventCorruption) << "Failed to find the event history tree.\n";
      }
    }
  }

  void RootFile::initializeDuplicateChecker(
      std::vector<std::shared_ptr<IndexIntoFile>> const& indexesIntoFiles,
      std::vector<std::shared_ptr<IndexIntoFile>>::size_type currentIndexIntoFile) {
    if (duplicateChecker_ && !duplicateChecker_->checkDisabled()) {
      if (eventTree_.next()) {
        auto const evtAux = fillThisEventAuxiliary();

        duplicateChecker_->inputFileOpened(evtAux.isRealData(), indexIntoFile_, indexesIntoFiles, currentIndexIntoFile);
      }
      eventTree_.setEntryNumber(IndexIntoFile::invalidEntry);
    }
  }

  void RootFile::setPresenceInProductRegistry(ProductRegistry& inputProdDescReg,
                                              StoredProcessBlockHelper const& storedProcessBlockHelper) {
    // Set product presence information in the product registry.
    // "Presence" is a boolean that is true if and only if the TBranch exists
    // in the TTree (except it will be false for ProcessBlock products in non-Primary
    // input files).
    ProductRegistry::ProductList& pList = inputProdDescReg.productListUpdator();
    for (auto& product : pList) {
      BranchDescription& prod = product.second;
      prod.init();
      if (prod.branchType() == InProcess) {
        std::vector<std::string> const& processes = storedProcessBlockHelper.processesWithProcessBlockProducts();
        auto it = std::find(processes.begin(), processes.end(), prod.processName());
        if (it != processes.end()) {
          auto index = std::distance(processes.begin(), it);
          processBlockTrees_[index]->setPresence(prod, newBranchToOldBranch(prod.branchName()));
        } else {
          // Given current rules for saving BranchDescriptions, this case should only occur
          // in non-Primary sequences.
          prod.setDropped(true);
        }
      } else {
        treePointers_[prod.branchType()]->setPresence(prod, newBranchToOldBranch(prod.branchName()));
      }
    }
  }

  void RootFile::markBranchToBeDropped(bool dropDescendants,
                                       BranchDescription const& branch,
                                       std::set<BranchID>& branchesToDrop,
                                       std::map<BranchID, BranchID> const& droppedToKeptAlias) const {
    if (dropDescendants) {
      branchChildren_->appendToDescendants(branch, branchesToDrop, droppedToKeptAlias);
    } else {
      branchesToDrop.insert(branch.branchID());
    }
  }

  void RootFile::dropOnInputAndReorder(ProductRegistry& reg,
                                       ProductSelectorRules const& rules,
                                       bool dropDescendants,
                                       InputType inputType,
                                       StoredProcessBlockHelper& storedProcessBlockHelper,
                                       ProcessBlockHelper const* processBlockHelper) {
    ProductRegistry::ProductList& prodList = reg.productListUpdator();

    // First fill in a map we will need to navigate to descendants
    // in the case of EDAliases.
    std::map<BranchID, BranchID> droppedToKeptAlias;
    for (auto const& product : prodList) {
      BranchDescription const& prod = product.second;
      if (prod.branchID() != prod.originalBranchID() && prod.present()) {
        droppedToKeptAlias[prod.originalBranchID()] = prod.branchID();
      }
    }

    // This object will select products based on the branchName and the
    // keep and drop statements which are in the source configuration.
    ProductSelector productSelector;
    productSelector.initialize(rules, reg.allBranchDescriptions());

    // In this pass, fill in a set of branches to be dropped.
    // Don't drop anything yet.
    std::set<BranchID> branchesToDrop;
    std::vector<BranchDescription const*> associationDescriptions;
    for (auto const& product : prodList) {
      BranchDescription const& prod = product.second;
      if (inputType != InputType::Primary && prod.branchType() == InProcess) {
        markBranchToBeDropped(dropDescendants, prod, branchesToDrop, droppedToKeptAlias);
      } else if (prod.unwrappedType() == typeid(ThinnedAssociation) && prod.present()) {
        // Special handling for ThinnedAssociations
        if (inputType != InputType::SecondarySource) {
          associationDescriptions.push_back(&prod);
        } else {
          markBranchToBeDropped(dropDescendants, prod, branchesToDrop, droppedToKeptAlias);
        }
      } else if (!productSelector.selected(prod)) {
        markBranchToBeDropped(dropDescendants, prod, branchesToDrop, droppedToKeptAlias);
      }
    }

    if (inputType != InputType::SecondarySource) {
      // Decide whether to keep the thinned associations and corresponding
      // entries in the helper. For secondary source they are all dropped,
      // but in other cases we look for thinned collections the associations
      // redirect a Ref or Ptr to when dereferencing them.

      // Need a list of kept products in order to determine which thinned associations
      // are kept.
      std::set<BranchID> keptProductsInEvent;
      for (auto const& product : prodList) {
        BranchDescription const& prod = product.second;
        if (branchesToDrop.find(prod.branchID()) == branchesToDrop.end() && prod.present() &&
            prod.branchType() == InEvent) {
          keptProductsInEvent.insert(prod.branchID());
        }
      }

      // Decide which ThinnedAssociations to keep and store the decision in keepAssociation
      std::map<BranchID, bool> keepAssociation;
      fileThinnedAssociationsHelper_->selectAssociationProducts(
          associationDescriptions, keptProductsInEvent, keepAssociation);

      for (auto association : associationDescriptions) {
        if (!keepAssociation[association->branchID()]) {
          markBranchToBeDropped(dropDescendants, *association, branchesToDrop, droppedToKeptAlias);
        }
      }

      // Also delete the dropped associations from the ThinnedAssociationsHelper
      auto temp = std::make_unique<ThinnedAssociationsHelper>();
      for (auto const& associationBranches : fileThinnedAssociationsHelper_->data()) {
        auto iter = keepAssociation.find(associationBranches.association());
        if (iter != keepAssociation.end() && iter->second) {
          temp->addAssociation(associationBranches);
        }
      }
      // propagate_const<T> has no reset() function
      fileThinnedAssociationsHelper_ = std::unique_ptr<ThinnedAssociationsHelper>(temp.release());
    }

    // On this pass, actually drop the branches.
    std::set<std::string> processesWithKeptProcessBlockProducts;
    std::set<BranchID>::const_iterator branchesToDropEnd = branchesToDrop.end();
    for (ProductRegistry::ProductList::iterator it = prodList.begin(), itEnd = prodList.end(); it != itEnd;) {
      BranchDescription const& prod = it->second;
      bool drop = branchesToDrop.find(prod.branchID()) != branchesToDropEnd;
      if (drop) {
        if (!prod.dropped()) {
          if (productSelector.selected(prod) && prod.unwrappedType() != typeid(ThinnedAssociation) &&
              prod.branchType() != InProcess) {
            LogWarning("RootFile") << "Branch '" << prod.branchName() << "' is being dropped from the input\n"
                                   << "of file '" << file_ << "' because it is dependent on a branch\n"
                                   << "that was explicitly dropped.\n";
          }
          if (prod.branchType() == InProcess) {
            std::vector<std::string> const& processes = storedProcessBlockHelper.processesWithProcessBlockProducts();
            auto it = std::find(processes.begin(), processes.end(), prod.processName());
            assert(it != processes.end());
            auto index = std::distance(processes.begin(), it);
            processBlockTrees_[index]->dropBranch(newBranchToOldBranch(prod.branchName()));
          } else {
            treePointers_[prod.branchType()]->dropBranch(newBranchToOldBranch(prod.branchName()));
          }
          hasNewlyDroppedBranch_[prod.branchType()] = true;
        }
        ProductRegistry::ProductList::iterator icopy = it;
        ++it;
        prodList.erase(icopy);
      } else {
        if (prod.branchType() == InProcess && prod.present()) {
          processesWithKeptProcessBlockProducts.insert(prod.processName());
        }
        ++it;
      }
    }

    dropProcessesAndReorder(storedProcessBlockHelper, processesWithKeptProcessBlockProducts, processBlockHelper);

    // Drop on input mergeable run and lumi products, this needs to be invoked for secondary file input
    if (inputType == InputType::SecondaryFile) {
      TString tString;
      for (ProductRegistry::ProductList::iterator it = prodList.begin(), itEnd = prodList.end(); it != itEnd;) {
        BranchDescription const& prod = it->second;
        if (prod.branchType() != InEvent && prod.branchType() != InProcess) {
          TClass* cp = prod.wrappedType().getClass();
          void* p = cp->New();
          int offset = cp->GetBaseClassOffset(edProductClass_);
          std::unique_ptr<WrapperBase> edp = getWrapperBasePtr(p, offset);
          if (edp->isMergeable()) {
            treePointers_[prod.branchType()]->dropBranch(newBranchToOldBranch(prod.branchName()));
            ProductRegistry::ProductList::iterator icopy = it;
            ++it;
            prodList.erase(icopy);
          } else {
            ++it;
          }
        } else
          ++it;
      }
    }
  }

  void RootFile::dropProcessesAndReorder(StoredProcessBlockHelper& storedProcessBlockHelper,
                                         std::set<std::string> const& processesWithKeptProcessBlockProducts,
                                         ProcessBlockHelper const* processBlockHelper) {
    // Modify storedProcessBlockHelper and processBlockTrees_
    // This should account for dropOnInput and also make the
    // order of process blocks in input files after the first
    // be the same as the first. Processes with no ProcessBlock
    // products should be removed. After this executes,
    // the items in storedProcessBlockHelper
    // and processBlockTrees should be in exact one to one
    // correspondence and in the same order. For input files
    // after the first, these items should be either the same
    // as or a subset of the items in processBlockHelper and in
    // the same order.

    if (processBlockTrees_.empty()) {
      return;
    }

    std::vector<unsigned int> nEntries;
    nEntries.reserve(processBlockTrees_.size());
    for (auto const& processBlockTree : processBlockTrees_) {
      nEntries.push_back(processBlockTree->entries());
    }

    bool firstInputFile = !processBlockHelper->initializedFromInput();
    bool isModified = false;
    std::vector<unsigned int> finalIndexToStoredIndex;

    if (firstInputFile) {
      isModified = processBlockHelper->firstFileDropProcessesAndReorderStored(
          storedProcessBlockHelper, processesWithKeptProcessBlockProducts, nEntries, finalIndexToStoredIndex);
    } else {
      isModified =
          processBlockHelper->dropProcessesAndReorderStored(storedProcessBlockHelper,
                                                            processesWithKeptProcessBlockProducts,
                                                            nEntries,
                                                            finalIndexToStoredIndex,
                                                            processBlockHelper->processesWithProcessBlockProducts());
    }

    // At this point, any modifications to storedProcessBlockHelper are done.
    // Make consistent changes to processBlockTrees_ and this will cause
    // unneeded RootTrees to be deleted.
    if (isModified) {
      std::vector<edm::propagate_const<std::unique_ptr<RootTree>>> newProcessBlockTrees;
      unsigned int nFinalProducts = storedProcessBlockHelper.processesWithProcessBlockProducts().size();
      for (unsigned int j = 0; j < nFinalProducts; ++j) {
        unsigned int iStored = finalIndexToStoredIndex[j];
        newProcessBlockTrees.push_back(std::move(processBlockTrees_[iStored]));
      }
      processBlockTrees_.swap(newProcessBlockTrees);
    }
  }

  void RootFile::setSignals(
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadSource,
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadSource) {
    eventTree_.setSignals(preEventReadSource, postEventReadSource);
  }

  void RootFile::makeProcessBlockRootTrees(std::shared_ptr<InputFile> filePtr,
                                           int treeMaxVirtualSize,
                                           bool enablePrefetching,
                                           InputType inputType,
                                           StoredProcessBlockHelper const& storedProcessBlockHelper) {
    // When this functions returns there will be exactly a 1-to-1 correspondence between the
    // processes listed in storedProcessBlockHelper and the RootTree objects created. processBlockTrees_
    // has pointers to the RootTree's and will be filled in the same order. The RootTree constructor
    // will throw an exception if one of these TTree's is not in the file and this should be all of
    // the ProcessBlock TTree's in the file. (later in the RootFile constructor, dropOnInput might
    // remove some and also reordering may occur).
    for (auto const& process : storedProcessBlockHelper.processesWithProcessBlockProducts()) {
      processBlockTrees_.emplace_back(std::make_unique<RootTree>(filePtr,
                                                                 InProcess,
                                                                 process,
                                                                 1,
                                                                 treeMaxVirtualSize,
                                                                 roottree::defaultNonEventCacheSize,
                                                                 roottree::defaultNonEventLearningEntries,
                                                                 enablePrefetching,
                                                                 inputType));
    }
  }

  std::unique_ptr<MakeProvenanceReader> RootFile::makeProvenanceReaderMaker(InputType inputType) {
    if (fileFormatVersion_.storedProductProvenanceUsed()) {
      readParentageTree(inputType);
      return std::make_unique<MakeReducedProvenanceReader>(parentageIDLookup_);
    } else if (fileFormatVersion_.splitProductIDs()) {
      readParentageTree(inputType);
      return std::make_unique<MakeFullProvenanceReader>();
    } else if (fileFormatVersion_.perEventProductIDs()) {
      auto entryDescriptionMap = std::make_unique<EntryDescriptionMap>();
      readEntryDescriptionTree(*entryDescriptionMap, inputType);
      return std::make_unique<MakeOldProvenanceReader>(std::move(entryDescriptionMap));
    } else {
      return std::make_unique<MakeDummyProvenanceReader>();
    }
  }

  std::shared_ptr<ProductProvenanceRetriever> RootFile::makeProductProvenanceRetriever(unsigned int iStreamID) {
    if (eventProductProvenanceRetrievers_.size() <= iStreamID) {
      eventProductProvenanceRetrievers_.resize(iStreamID + 1);
    }
    if (!eventProductProvenanceRetrievers_[iStreamID]) {
      // propagate_const<T> has no reset() function
      eventProductProvenanceRetrievers_[iStreamID] = std::make_shared<ProductProvenanceRetriever>(
          provenanceReaderMaker_->makeReader(eventTree_, daqProvenanceHelper_.get()));
    }
    eventProductProvenanceRetrievers_[iStreamID]->reset();
    return eventProductProvenanceRetriever(iStreamID);
  }

  class ReducedProvenanceReader : public ProvenanceReaderBase {
  public:
    ReducedProvenanceReader(RootTree* iRootTree,
                            std::vector<ParentageID> const& iParentageIDLookup,
                            DaqProvenanceHelper const* daqProvenanceHelper);

    std::set<ProductProvenance> readProvenance(unsigned int) const override;

  private:
    void readProvenanceAsync(WaitingTaskHolder task,
                             ModuleCallingContext const* moduleCallingContext,
                             unsigned int transitionIndex,
                             std::atomic<const std::set<ProductProvenance>*>& writeTo) const override;

    edm::propagate_const<RootTree*> rootTree_;
    edm::propagate_const<TBranch*> provBranch_;
    StoredProductProvenanceVector provVector_;
    StoredProductProvenanceVector const* pProvVector_;
    std::vector<ParentageID> const& parentageIDLookup_;
    DaqProvenanceHelper const* daqProvenanceHelper_;
    std::shared_ptr<std::recursive_mutex> mutex_;
    SharedResourcesAcquirer acquirer_;
  };

  ReducedProvenanceReader::ReducedProvenanceReader(RootTree* iRootTree,
                                                   std::vector<ParentageID> const& iParentageIDLookup,
                                                   DaqProvenanceHelper const* daqProvenanceHelper)
      : ProvenanceReaderBase(),
        rootTree_(iRootTree),
        pProvVector_(&provVector_),
        parentageIDLookup_(iParentageIDLookup),
        daqProvenanceHelper_(daqProvenanceHelper),
        mutex_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().second),
        acquirer_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().first) {
    provBranch_ =
        rootTree_->tree()->GetBranch(BranchTypeToProductProvenanceBranchName(rootTree_->branchType()).c_str());
  }

  namespace {
    using SignalType = signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)>;
    template <typename R>
    void readProvenanceAsyncImpl(R const* iThis,
                                 SerialTaskQueueChain& chain,
                                 WaitingTaskHolder task,
                                 unsigned int transitionIndex,
                                 std::atomic<const std::set<ProductProvenance>*>& writeTo,
                                 ModuleCallingContext const* iContext,
                                 SignalType const* pre,
                                 SignalType const* post) {
      if (nullptr == writeTo.load()) {
        //need to be sure the task isn't run until after the read
        WaitingTaskHolder taskHolder{task};
        auto pWriteTo = &writeTo;

        auto serviceToken = ServiceRegistry::instance().presentToken();

        chain.push(
            *taskHolder.group(),
            [holder = std::move(taskHolder),
             pWriteTo,
             iThis,
             transitionIndex,
             iContext,
             pre,
             post,
             serviceToken]() mutable {
              if (nullptr == pWriteTo->load()) {
                ServiceRegistry::Operate operate(serviceToken);
                std::unique_ptr<const std::set<ProductProvenance>> prov;
                try {
                  if (pre) {
                    pre->emit(*(iContext->getStreamContext()), *iContext);
                  }
                  prov = std::make_unique<const std::set<ProductProvenance>>(iThis->readProvenance(transitionIndex));
                  if (post) {
                    post->emit(*(iContext->getStreamContext()), *iContext);
                  }

                } catch (...) {
                  if (post) {
                    post->emit(*(iContext->getStreamContext()), *iContext);
                  }

                  holder.doneWaiting(std::current_exception());
                  return;
                }
                const std::set<ProductProvenance>* expected = nullptr;

                if (pWriteTo->compare_exchange_strong(expected, prov.get())) {
                  prov.release();
                }
              }
              holder.doneWaiting(std::exception_ptr());
            });
      }
    }
  }  // namespace

  void ReducedProvenanceReader::readProvenanceAsync(WaitingTaskHolder task,
                                                    ModuleCallingContext const* moduleCallingContext,
                                                    unsigned int transitionIndex,
                                                    std::atomic<const std::set<ProductProvenance>*>& writeTo) const {
    readProvenanceAsyncImpl(this,
                            acquirer_.serialQueueChain(),
                            task,
                            transitionIndex,
                            writeTo,
                            moduleCallingContext,
                            rootTree_->rootDelayedReader()->preEventReadFromSourceSignal(),
                            rootTree_->rootDelayedReader()->postEventReadFromSourceSignal());
  }

  std::set<ProductProvenance> ReducedProvenanceReader::readProvenance(unsigned int transitionIndex) const {
    {
      std::lock_guard<std::recursive_mutex> guard(*mutex_);
      ReducedProvenanceReader* me = const_cast<ReducedProvenanceReader*>(this);
      me->rootTree_->fillBranchEntry(
          me->provBranch_, me->rootTree_->entryNumberForIndex(transitionIndex), me->pProvVector_);
    }
    std::set<ProductProvenance> retValue;
    if (daqProvenanceHelper_) {
      for (auto const& prov : provVector_) {
        BranchID bid(prov.branchID_);
        retValue.emplace(daqProvenanceHelper_->mapBranchID(BranchID(prov.branchID_)),
                         daqProvenanceHelper_->mapParentageID(parentageIDLookup_[prov.parentageIDIndex_]));
      }
    } else {
      for (auto const& prov : provVector_) {
        if (prov.parentageIDIndex_ >= parentageIDLookup_.size()) {
          throw edm::Exception(errors::LogicError)
              << "ReducedProvenanceReader::ReadProvenance\n"
              << "The parentage ID index value " << prov.parentageIDIndex_
              << " is out of bounds.  The maximum value is " << parentageIDLookup_.size() - 1 << ".\n"
              << "This should never happen.\n"
              << "Please report this to the framework hypernews forum 'hn-cms-edmFramework@cern.ch'.\n";
        }
        retValue.emplace(BranchID(prov.branchID_), parentageIDLookup_[prov.parentageIDIndex_]);
      }
    }
    return retValue;
  }

  class FullProvenanceReader : public ProvenanceReaderBase {
  public:
    explicit FullProvenanceReader(RootTree* rootTree, DaqProvenanceHelper const* daqProvenanceHelper);
    ~FullProvenanceReader() override {}
    std::set<ProductProvenance> readProvenance(unsigned int transitionIndex) const override;

  private:
    void readProvenanceAsync(WaitingTaskHolder task,
                             ModuleCallingContext const* moduleCallingContext,
                             unsigned int transitionIndex,
                             std::atomic<const std::set<ProductProvenance>*>& writeTo) const override;

    RootTree* rootTree_;
    ProductProvenanceVector infoVector_;
    //All access to a ROOT file is serialized
    CMS_SA_ALLOW mutable ProductProvenanceVector* pInfoVector_;
    DaqProvenanceHelper const* daqProvenanceHelper_;
    std::shared_ptr<std::recursive_mutex> mutex_;
    SharedResourcesAcquirer acquirer_;
  };

  FullProvenanceReader::FullProvenanceReader(RootTree* rootTree, DaqProvenanceHelper const* daqProvenanceHelper)
      : ProvenanceReaderBase(),
        rootTree_(rootTree),
        infoVector_(),
        pInfoVector_(&infoVector_),
        daqProvenanceHelper_(daqProvenanceHelper),
        mutex_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().second),
        acquirer_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().first) {}

  void FullProvenanceReader::readProvenanceAsync(WaitingTaskHolder task,
                                                 ModuleCallingContext const* moduleCallingContext,
                                                 unsigned int transitionIndex,
                                                 std::atomic<const std::set<ProductProvenance>*>& writeTo) const {
    readProvenanceAsyncImpl(this,
                            acquirer_.serialQueueChain(),
                            task,
                            transitionIndex,
                            writeTo,
                            moduleCallingContext,
                            rootTree_->rootDelayedReader()->preEventReadFromSourceSignal(),
                            rootTree_->rootDelayedReader()->postEventReadFromSourceSignal());
  }

  std::set<ProductProvenance> FullProvenanceReader::readProvenance(unsigned int transitionIndex) const {
    {
      std::lock_guard<std::recursive_mutex> guard(*mutex_);
      rootTree_->fillBranchEntryMeta(
          rootTree_->branchEntryInfoBranch(), rootTree_->entryNumberForIndex(transitionIndex), pInfoVector_);
    }
    std::set<ProductProvenance> retValue;
    if (daqProvenanceHelper_) {
      for (auto const& info : infoVector_) {
        retValue.emplace(daqProvenanceHelper_->mapBranchID(info.branchID()),
                         daqProvenanceHelper_->mapParentageID(info.parentageID()));
      }
    } else {
      for (auto const& info : infoVector_) {
        retValue.emplace(info);
      }
    }
    return retValue;
  }

  class OldProvenanceReader : public ProvenanceReaderBase {
  public:
    explicit OldProvenanceReader(RootTree* rootTree,
                                 EntryDescriptionMap const& theMap,
                                 DaqProvenanceHelper const* daqProvenanceHelper);
    ~OldProvenanceReader() override {}
    std::set<ProductProvenance> readProvenance(unsigned int transitionIndex) const override;

  private:
    void readProvenanceAsync(WaitingTaskHolder task,
                             ModuleCallingContext const* moduleCallingContext,
                             unsigned int transitionIndex,
                             std::atomic<const std::set<ProductProvenance>*>& writeTo) const override;

    edm::propagate_const<RootTree*> rootTree_;
    std::vector<EventEntryInfo> infoVector_;
    //All access to ROOT file are serialized
    CMS_SA_ALLOW mutable std::vector<EventEntryInfo>* pInfoVector_;
    EntryDescriptionMap const& entryDescriptionMap_;
    DaqProvenanceHelper const* daqProvenanceHelper_;
    std::shared_ptr<std::recursive_mutex> mutex_;
    SharedResourcesAcquirer acquirer_;
  };

  OldProvenanceReader::OldProvenanceReader(RootTree* rootTree,
                                           EntryDescriptionMap const& theMap,
                                           DaqProvenanceHelper const* daqProvenanceHelper)
      : ProvenanceReaderBase(),
        rootTree_(rootTree),
        infoVector_(),
        pInfoVector_(&infoVector_),
        entryDescriptionMap_(theMap),
        daqProvenanceHelper_(daqProvenanceHelper),
        mutex_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().second),
        acquirer_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().first) {}

  void OldProvenanceReader::readProvenanceAsync(WaitingTaskHolder task,
                                                ModuleCallingContext const* moduleCallingContext,
                                                unsigned int transitionIndex,
                                                std::atomic<const std::set<ProductProvenance>*>& writeTo) const {
    readProvenanceAsyncImpl(this,
                            acquirer_.serialQueueChain(),
                            task,
                            transitionIndex,
                            writeTo,
                            moduleCallingContext,
                            rootTree_->rootDelayedReader()->preEventReadFromSourceSignal(),
                            rootTree_->rootDelayedReader()->postEventReadFromSourceSignal());
  }

  std::set<ProductProvenance> OldProvenanceReader::readProvenance(unsigned int transitionIndex) const {
    {
      std::lock_guard<std::recursive_mutex> guard(*mutex_);
      rootTree_->branchEntryInfoBranch()->SetAddress(&pInfoVector_);
      roottree::getEntry(rootTree_->branchEntryInfoBranch(), rootTree_->entryNumberForIndex(transitionIndex));
    }
    std::set<ProductProvenance> retValue;
    for (auto const& info : infoVector_) {
      EntryDescriptionMap::const_iterator iter = entryDescriptionMap_.find(info.entryDescriptionID());
      assert(iter != entryDescriptionMap_.end());
      Parentage parentage(iter->second.parents());
      if (daqProvenanceHelper_) {
        retValue.emplace(daqProvenanceHelper_->mapBranchID(info.branchID()),
                         daqProvenanceHelper_->mapParentageID(parentage.id()));
      } else {
        retValue.emplace(info.branchID(), parentage.id());
      }
    }
    return retValue;
  }

  class DummyProvenanceReader : public ProvenanceReaderBase {
  public:
    DummyProvenanceReader();
    ~DummyProvenanceReader() override {}

  private:
    std::set<ProductProvenance> readProvenance(unsigned int) const override;
    void readProvenanceAsync(WaitingTaskHolder task,
                             ModuleCallingContext const* moduleCallingContext,
                             unsigned int transitionIndex,
                             std::atomic<const std::set<ProductProvenance>*>& writeTo) const override;
  };

  DummyProvenanceReader::DummyProvenanceReader() : ProvenanceReaderBase() {}

  std::set<ProductProvenance> DummyProvenanceReader::readProvenance(unsigned int) const {
    // Not providing parentage!!!
    return std::set<ProductProvenance>{};
  }
  void DummyProvenanceReader::readProvenanceAsync(WaitingTaskHolder task,
                                                  ModuleCallingContext const* moduleCallingContext,
                                                  unsigned int transitionIndex,
                                                  std::atomic<const std::set<ProductProvenance>*>& writeTo) const {
    if (nullptr == writeTo.load()) {
      auto emptyProv = std::make_unique<const std::set<ProductProvenance>>();
      const std::set<ProductProvenance>* expected = nullptr;
      if (writeTo.compare_exchange_strong(expected, emptyProv.get())) {
        emptyProv.release();
      }
    }
  }

  std::unique_ptr<ProvenanceReaderBase> MakeDummyProvenanceReader::makeReader(RootTree&,
                                                                              DaqProvenanceHelper const*) const {
    return std::make_unique<DummyProvenanceReader>();
  }

  std::unique_ptr<ProvenanceReaderBase> MakeOldProvenanceReader::makeReader(
      RootTree& rootTree, DaqProvenanceHelper const* daqProvenanceHelper) const {
    return std::make_unique<OldProvenanceReader>(&rootTree, *entryDescriptionMap_, daqProvenanceHelper);
  }

  std::unique_ptr<ProvenanceReaderBase> MakeFullProvenanceReader::makeReader(
      RootTree& rootTree, DaqProvenanceHelper const* daqProvenanceHelper) const {
    return std::make_unique<FullProvenanceReader>(&rootTree, daqProvenanceHelper);
  }

  std::unique_ptr<ProvenanceReaderBase> MakeReducedProvenanceReader::makeReader(
      RootTree& rootTree, DaqProvenanceHelper const* daqProvenanceHelper) const {
    return std::make_unique<ReducedProvenanceReader>(&rootTree, parentageIDLookup_, daqProvenanceHelper);
  }
}  // namespace edm
