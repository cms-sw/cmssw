#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/StoredProductProvenance.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/Utilities/interface/propagate_const.h"

#include "RNTupleInputFile.h"
#include "RNTupleDelayedReader.h"
#include <memory>

namespace edm {

  namespace {
    class ProvenanceReader : public edm::ProvenanceReaderBase {
    public:
      ProvenanceReader(edm::input::DataProductsRNTuple* iRNTuple,
                       std::vector<ParentageID> const* iParentageIDLookup,
                       std::vector<int>::const_iterator iStreamToEntryBegin,
                       SharedResourcesAcquirer* iAcquirer,
                       std::recursive_mutex* iMutex)
          : rntuple_(iRNTuple),
            acquirer_(iAcquirer),
            mutex_(iMutex),
            parentageIDLookup_(iParentageIDLookup),
            entryBegin_(iStreamToEntryBegin),
            descriptor_(rntuple_->findDescriptorID("EventProductProvenance")) {}

      std::set<ProductProvenance> readProvenance(unsigned int transitionIndex) const final;
      void readProvenanceAsync(WaitingTaskHolder task,
                               ModuleCallingContext const* moduleCallingContext,
                               unsigned int transitionIndex,
                               std::atomic<const std::set<ProductProvenance>*>& writeTo) const noexcept final;

    public:
      edm::input::DataProductsRNTuple* rntuple_;
      SharedResourcesAcquirer* acquirer_;
      std::recursive_mutex* mutex_;
      std::vector<ParentageID> const* parentageIDLookup_;
      std::vector<int>::const_iterator entryBegin_;
      ROOT::Experimental::DescriptorId_t descriptor_;
    };

    std::set<ProductProvenance> ProvenanceReader::readProvenance(unsigned int transitionIndex) const {
      std::vector<StoredProductProvenance> provVec;
      {
        std::shared_ptr<std::vector<StoredProductProvenance>> provPtr{&provVec, [](auto) {}};
        std::lock_guard<std::recursive_mutex> guard(*mutex_);
        auto v = rntuple_->viewFor(descriptor_, provPtr);
        auto entry = *(entryBegin_ + transitionIndex);
        v(entry);
      }

      std::set<ProductProvenance> retValue;
      for (auto const& prov : provVec) {
        if (prov.parentageIDIndex_ >= parentageIDLookup_->size()) {
          throw edm::Exception(errors::LogicError)
              << "RNTuple ProvenanceReader::ReadProvenance\n"
              << "The parentage ID index value " << prov.parentageIDIndex_
              << " is out of bounds.  The maximum value is " << parentageIDLookup_->size() - 1 << ".\n"
              << "This should never happen.\n"
              << "Please report this to the framework developers.";
        }
        retValue.emplace(BranchID(prov.branchID_), (*parentageIDLookup_)[prov.parentageIDIndex_]);
      }

      return retValue;
    }

    void ProvenanceReader::readProvenanceAsync(WaitingTaskHolder task,
                                               ModuleCallingContext const* moduleCallingContext,
                                               unsigned int transitionIndex,
                                               std::atomic<const std::set<ProductProvenance>*>& writeTo) const noexcept {
      if (nullptr == writeTo.load()) {
        //need to be sure the task isn't run until after the read
        WaitingTaskHolder taskHolder{task};
        auto pWriteTo = &writeTo;

        auto serviceToken = ServiceRegistry::instance().presentToken();

        acquirer_->serialQueueChain().push(
            *taskHolder.group(),
            [holder = std::move(taskHolder),
             pWriteTo,
             this,
             transitionIndex,
             moduleCallingContext,
             serviceToken]() mutable {
              if (nullptr == pWriteTo->load()) {
                ServiceRegistry::Operate operate(serviceToken);
                std::unique_ptr<const std::set<ProductProvenance>> prov;
                CMS_SA_ALLOW try {
                  prov = std::make_unique<const std::set<ProductProvenance>>(this->readProvenance(transitionIndex));
                } catch (...) {
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
  class RNTupleSource : public InputSource {
  public:
    explicit RNTupleSource(ParameterSet const& pset, InputSourceDescription const& desc);
    using InputSource::processHistoryRegistryForUpdate;
    using InputSource::productRegistryUpdate;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void skip(int offset) override;

    ItemTypeInfo getNextItemType() override;
    void readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal) override;
    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
    void readEvent_(EventPrincipal& eventPrincipal) override;
    std::shared_ptr<RunAuxiliary> readRunAuxiliary_() override;
    void readRun_(RunPrincipal& runPrincipal) override;
    std::shared_ptr<FileBlock> readFile_() override;
    void closeFile_() override;

    std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> resourceSharedWithDelayedReader_() override;

    std::unique_ptr<SharedResourcesAcquirer>
        resourceSharedWithDelayedReaderPtr_;  // We do not use propagate_const because the acquirer is itself mutable.
    std::shared_ptr<std::recursive_mutex> mutexSharedWithDelayedReader_;

    std::unique_ptr<RNTupleInputFile> file_;
    std::vector<input::RNTupleDelayedReader> runReaders_;
    std::vector<input::RNTupleDelayedReader> lumiReaders_;
    std::vector<input::RNTupleDelayedReader> eventReaders_;

    std::vector<ParentageID> parentageIDLookup_;
    std::vector<std::shared_ptr<ProductProvenanceRetriever>> provenanceRetrievers_;
    std::vector<int> entryForStream_;

    EventToProcessBlockIndexes processBlockIndexes_;

    ROOT::Experimental::DescriptorId_t eventSelectionsID_;
    ROOT::Experimental::DescriptorId_t eventProductProvenanceID_;
    ROOT::Experimental::DescriptorId_t eventBranchListIndexesID_;
    bool enableMetrics_ = false;
    bool useClusterCache_ = true;
    bool startedFirstFile_ = false;
  };

  RNTupleSource::RNTupleSource(ParameterSet const& pset, InputSourceDescription const& desc)
      : InputSource(pset, desc),
        entryForStream_(std::size_t(desc.allocations_->numberOfStreams()), int(0)),
        enableMetrics_(pset.getUntrackedParameter<bool>("enableMetrics")),
        useClusterCache_(pset.getUntrackedParameter<bool>("useClusterCache")) {
    auto resources = SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader();
    resourceSharedWithDelayedReaderPtr_ = std::make_unique<SharedResourcesAcquirer>(std::move(resources.first));
    mutexSharedWithDelayedReader_ = resources.second;

    RNTupleInputFile::Options ops;
    ops.enableMetrics_ = enableMetrics_;
    ops.useClusterCache_ = useClusterCache_;
    auto files = pset.getUntrackedParameter<std::vector<std::string>>("fileNames");
    if (files.empty()) {
      throw cms::Exception("NoInputFiles");
    }
    if (files.size() > 1) {
      throw edm::Exception(edm::errors::Configuration) << "RNTupleSource presently only support reading 1 file";
    }
    file_ = std::make_unique<RNTupleInputFile>(files[0], ops);

    BranchIDLists branchIDLists;
    file_->readMeta(productRegistryUpdate(), processHistoryRegistryForUpdate(), branchIDLists);
    branchIDListHelper()->updateFromInput(branchIDLists);
    file_->readParameterSets();

    runReaders_.reserve(desc.allocations_->numberOfRuns());
    for (unsigned int i = 0; i < desc.allocations_->numberOfRuns(); ++i) {
      runReaders_.emplace_back(
          file_->runProducts(), resourceSharedWithDelayedReaderPtr_.get(), mutexSharedWithDelayedReader_.get());
    }
    lumiReaders_.reserve(desc.allocations_->numberOfLuminosityBlocks());
    for (unsigned int i = 0; i < desc.allocations_->numberOfLuminosityBlocks(); ++i) {
      lumiReaders_.emplace_back(file_->luminosityBlockProducts(),
                                resourceSharedWithDelayedReaderPtr_.get(),
                                mutexSharedWithDelayedReader_.get());
    }
    eventReaders_.reserve(desc.allocations_->numberOfStreams());
    for (unsigned int i = 0; i < desc.allocations_->numberOfStreams(); ++i) {
      eventReaders_.emplace_back(
          file_->eventProducts(), resourceSharedWithDelayedReaderPtr_.get(), mutexSharedWithDelayedReader_.get());
    }
    {
      std::string kEventProvName = "EventProductProvenance";
      std::string kEventSelName = "EventSelections";
      std::string kBranchListName = "BranchListIndexes";

      eventSelectionsID_ = file_->eventProducts()->findDescriptorID(kEventSelName);
      eventProductProvenanceID_ = file_->eventProducts()->findDescriptorID(kEventProvName);
      eventBranchListIndexesID_ = file_->eventProducts()->findDescriptorID(kBranchListName);
    }

    parentageIDLookup_ = file_->readParentage();

    provenanceRetrievers_.reserve(desc.allocations_->numberOfStreams());

    for (unsigned int i = 0; i < desc.allocations_->numberOfStreams(); ++i) {
      provenanceRetrievers_.push_back(std::make_shared<ProductProvenanceRetriever>(
          std::make_unique<ProvenanceReader>(file_->eventProducts(),
                                             &parentageIDLookup_,
                                             entryForStream_.begin(),
                                             resourceSharedWithDelayedReaderPtr_.get(),
                                             mutexSharedWithDelayedReader_.get())));
    }
  }

  void RNTupleSource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.addUntracked<std::vector<std::string>>("fileNames")->setComment("only the first file will be used for now");
    desc.addUntracked<bool>("enableMetrics", false);
    desc.addUntracked<bool>("useClusterCache", true);

    //make interface compatible with PoolSource.
    desc.addOptionalUntracked<std::vector<std::string>>("secondaryFileNames");
    desc.addOptionalUntracked<bool>("needSecondaryFileNames");
    desc.addOptionalUntracked<std::string>("overrideCatalog");
    desc.addOptionalUntracked<bool>("skipBadFiles");
    desc.addOptionalUntracked<bool>("bypassVersionCheck");
    desc.addObsoleteUntracked<int>("treeMaxVirtualSize");
    desc.addOptionalUntracked<bool>("dropDescendantsOfDroppedBranches");
    desc.addOptionalUntracked<bool>("labelRawDataLikeMC");
    desc.addOptionalUntracked<bool>("delayReadingEventProducts");
    ProductSelectorRules::fillDescription(desc, "inputCommands");
    InputSource::fillDescription(desc);
    //RootPrimaryFileSequence::fillDescription(desc);
    //RunHelperBase::fillDescription(desc);

    descriptions.addDefault(desc);
  }

  void RNTupleSource::skip(int offset) { file_->skipEvents(offset); }

  InputSource::ItemTypeInfo RNTupleSource::getNextItemType() {
    if (not startedFirstFile_) {
      return InputSource::ItemType::IsFile;
    }
    auto entryType = file_->getNextItemType();
    switch (entryType) {
      case IndexIntoFile::kEnd:
        return InputSource::ItemType::IsStop;
      case IndexIntoFile::kRun:
        return InputSource::ItemType::IsRun;
      case IndexIntoFile::kLumi:
        return InputSource::ItemType::IsLumi;
      case IndexIntoFile::kEvent:
        return InputSource::ItemType::IsEvent;
      default:
        assert(false);
    }
    assert(false);
    return InputSource::ItemType::IsStop;
  }

  void RNTupleSource::readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal) {
    auto history = processHistoryRegistry().getMapped(lumiPrincipal.aux().processHistoryID());
    auto entry = file_->readLuminosityBlock();
    auto& reader = lumiReaders_[lumiPrincipal.index()];
    reader.setEntry(entry);
    lumiPrincipal.fillLuminosityBlockPrincipal(history, &reader);
    // Read in all the products now.
    lumiPrincipal.readAllFromSourceAndMergeImmediately();
    lumiPrincipal.setShouldWriteLumi(LuminosityBlockPrincipal::kYes);
  }
  std::shared_ptr<LuminosityBlockAuxiliary> RNTupleSource::readLuminosityBlockAuxiliary_() {
    return file_->readLuminosityBlockAuxiliary();
  }
  void RNTupleSource::readEvent_(EventPrincipal& eventPrincipal) {
    auto aux = file_->readEventAuxiliary();

    auto history = processHistoryRegistry().getMapped(aux->processHistoryID());
    auto entry = file_->readEvent();
    entryForStream_[eventPrincipal.streamID().value()] = entry;
    auto& reader = eventReaders_[eventPrincipal.streamID().value()];
    reader.setEntry(entry);

    EventSelectionIDVector esids;
    BranchListIndexes bli;

    {
      std::shared_ptr<EventSelectionIDVector> pEsids{&esids, [](auto) {}};
      auto v = file_->eventProducts()->viewFor(eventSelectionsID_, pEsids);
      v(entry);
    }
    {
      std::shared_ptr<BranchListIndexes> pBli{&bli, [](auto) {}};
      auto v = file_->eventProducts()->viewFor(eventBranchListIndexesID_, pBli);
      v(entry);
    }
    branchIDListHelper()->fixBranchListIndexes(bli);

    eventPrincipal.fillEventPrincipal(*aux,
                                      history,
                                      std::move(esids),
                                      std::move(bli),
                                      processBlockIndexes_,
                                      *provenanceRetrievers_[eventPrincipal.streamID().value()],
                                      &reader);
  }

  std::shared_ptr<RunAuxiliary> RNTupleSource::readRunAuxiliary_() { return file_->readRunAuxiliary(); }
  void RNTupleSource::readRun_(RunPrincipal& runPrincipal) {
    auto entry = file_->readRun();
    auto& reader = runReaders_[runPrincipal.index()];
    reader.setEntry(entry);
    runPrincipal.fillRunPrincipal(processHistoryRegistry(), &reader);
    // Read in all the products now.
    runPrincipal.readAllFromSourceAndMergeImmediately();
    //In the future, when this code can skip runs we will need to
    // set this value.
    runPrincipal.setShouldWriteRun(RunPrincipal::kYes);
  }

  std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> RNTupleSource::resourceSharedWithDelayedReader_() {
    return std::make_pair(resourceSharedWithDelayedReaderPtr_.get(), mutexSharedWithDelayedReader_.get());
  }

  std::shared_ptr<edm::FileBlock> RNTupleSource::readFile_() {
    startedFirstFile_ = true;
    return std::make_shared<edm::FileBlock>();
  }

  void RNTupleSource::closeFile_() {
    if (enableMetrics_) {
      file_->printInfoForEvent(std::cout);
    }
  }

}  // namespace edm

using edm::RNTupleSource;
DEFINE_FWK_INPUT_SOURCE(RNTupleSource);
