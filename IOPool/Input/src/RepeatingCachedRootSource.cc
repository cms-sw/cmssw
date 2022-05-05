// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     RepeatingCachedRootSource
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Mon, 15 Mar 2021 19:02:31 GMT
//

// system include files
#include <memory>

// user include files
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/EventToProcessBlockIndexes.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

#include "IOPool/Common/interface/RootServiceChecker.h"

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"

#include "FWCore/Framework/interface/InputSourceMacros.h"

#include "RunHelper.h"
#include "RootFile.h"
#include "InputFile.h"
#include "DuplicateChecker.h"

namespace edm {
  class RunHelperBase;

  class RepeatingCachedRootSource : public InputSource {
  public:
    RepeatingCachedRootSource(ParameterSet const& pset, InputSourceDescription const& desc);

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

    std::shared_ptr<WrapperBase> getProduct(unsigned int iStreamIndex,
                                            BranchID const& k,
                                            EDProductGetter const* ep) const;

    class RCProductGetter : public EDProductGetter {
    public:
      RCProductGetter(RCProductGetter const& iOther) : map_(iOther.map_), wrappers_(iOther.wrappers_) {}

      RCProductGetter const& operator=(RCProductGetter const& iOther) {
        map_ = iOther.map_;
        wrappers_ = iOther.wrappers_;
        return *this;
      }

      RCProductGetter(std::map<edm::ProductID, size_t> const* iMap,
                      std::vector<std::shared_ptr<edm::WrapperBase>> const* iWrappers)
          : map_(iMap), wrappers_(iWrappers) {}

      WrapperBase const* getIt(ProductID const&) const override;

      std::optional<std::tuple<WrapperBase const*, unsigned int>> getThinnedProduct(ProductID const&,
                                                                                    unsigned int key) const override;

      void getThinnedProducts(ProductID const& pid,
                              std::vector<WrapperBase const*>& foundContainers,
                              std::vector<unsigned int>& keys) const override;

      OptionalThinnedKey getThinnedKeyFrom(ProductID const& parent,
                                           unsigned int key,
                                           ProductID const& thinned) const override;

    private:
      unsigned int transitionIndex_() const override;

      std::map<edm::ProductID, size_t> const* map_;
      std::vector<std::shared_ptr<edm::WrapperBase>> const* wrappers_;
    };

    class RCDelayedReader : public edm::DelayedReader {
    public:
      std::shared_ptr<edm::WrapperBase> getProduct_(edm::BranchID const& k, edm::EDProductGetter const* ep) final {
        return m_source->getProduct(m_streamIndex, k, ep);
      }
      void mergeReaders_(edm::DelayedReader*) final { assert(false); }
      void reset_() final {}

      unsigned int m_streamIndex;
      edm::RepeatingCachedRootSource const* m_source;

      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadFromSourceSignal()
          const final {
        return nullptr;
      }
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadFromSourceSignal()
          const final {
        return nullptr;
      }
    };

  protected:
    ItemType getNextItemType() override;
    void readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal) override;
    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
    void readEvent_(EventPrincipal& eventPrincipal) override;

  private:
    std::shared_ptr<RunAuxiliary> readRunAuxiliary_() override;
    void readRun_(RunPrincipal& runPrincipal) override;
    bool readIt(EventID const& id, EventPrincipal& eventPrincipal, StreamContext& streamContext) override;
    void skip(int offset) override;
    bool goToEvent_(EventID const& eventID) override;
    void beginJob() override;

    void fillProcessBlockHelper_() override;
    bool nextProcessBlock_(ProcessBlockPrincipal&) override;
    void readProcessBlock_(ProcessBlockPrincipal&) override;

    std::unique_ptr<RootFile> makeRootFile(std::string const& logicalFileName,
                                           std::string const& pName,
                                           bool isSkipping,
                                           std::shared_ptr<InputFile> filePtr,
                                           std::shared_ptr<EventSkipperByID> skipper,
                                           std::shared_ptr<DuplicateChecker> duplicateChecker,
                                           std::vector<std::shared_ptr<IndexIntoFile>>& indexesIntoFiles);

    RootServiceChecker rootServiceChecker_;
    ProductSelectorRules selectorRules_;
    edm::propagate_const<std::unique_ptr<RunHelperBase>> runHelper_;
    std::unique_ptr<RootFile> rootFile_;
    std::vector<ProcessHistoryID> orderedProcessHistoryIDs_;
    std::vector<std::vector<std::shared_ptr<edm::WrapperBase>>> cachedWrappers_;
    std::vector<RCProductGetter> getters_;  //one per cached event
    std::vector<EventAuxiliary> eventAuxs_;
    EventSelectionIDVector selectionIDs_;
    BranchListIndexes branchListIndexes_;
    ProductProvenanceRetriever provRetriever_;
    std::vector<RCDelayedReader> delayedReaders_;  //one per stream
    std::map<edm::BranchID, size_t> branchIDToWrapperIndex_;
    std::map<edm::ProductID, size_t> productIDToWrapperIndex_;
    std::vector<size_t> streamToCacheIndex_;
    size_t nextEventIndex_ = 0;
    ItemType presentState_ = IsFile;
    unsigned long long eventIndex_ = 0;
  };
}  // namespace edm

using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RepeatingCachedRootSource::RepeatingCachedRootSource(ParameterSet const& pset, InputSourceDescription const& desc)
    : InputSource(pset, desc),
      selectorRules_(pset, "inputCommands", "InputSource"),
      runHelper_(std::make_unique<DefaultRunHelper>()),
      cachedWrappers_(pset.getUntrackedParameter<unsigned int>("repeatNEvents")),
      eventAuxs_(cachedWrappers_.size()),
      provRetriever_(0),
      delayedReaders_(desc.allocations_->numberOfStreams()),
      streamToCacheIndex_(desc.allocations_->numberOfStreams(), 0) {
  {
    getters_.reserve(cachedWrappers_.size());
    for (auto& cw : cachedWrappers_) {
      getters_.emplace_back(&productIDToWrapperIndex_, &cw);
    }

    int index = 0;
    std::for_each(delayedReaders_.begin(), delayedReaders_.end(), [&index, this](auto& iR) {
      iR.m_streamIndex = index++;
      iR.m_source = this;
    });
  }
  auto logicalFileName = pset.getUntrackedParameter<std::string>("fileName");
  InputFileCatalog catalog(std::vector<std::string>(1U, logicalFileName), "");
  auto const& physicalFileName = catalog.fileCatalogItems().front().fileNames().front();
  auto const nEventsToSkip = pset.getUntrackedParameter<unsigned int>("skipEvents");
  std::shared_ptr<EventSkipperByID> skipper(EventSkipperByID::create(pset).release());

  auto duplicateChecker = std::make_shared<DuplicateChecker>(pset);

  std::vector<std::shared_ptr<IndexIntoFile>> indexesIntoFiles(1);

  auto input =
      std::make_shared<InputFile>(physicalFileName.c_str(), "  Initiating request to open file ", InputType::Primary);
  rootFile_ = makeRootFile(
      logicalFileName, physicalFileName, 0 != nEventsToSkip, input, skipper, duplicateChecker, indexesIntoFiles);
  rootFile_->reportOpened("repeating");

  auto const& prodList = rootFile_->productRegistry()->productList();
  productRegistryUpdate().updateFromInput(prodList);

  //setup caching
  auto nProdsInEvent =
      std::count_if(prodList.begin(), prodList.end(), [](auto&& iV) { return iV.second.branchType() == edm::InEvent; });
  {
    size_t index = 0;
    for (auto& prod : prodList) {
      if (prod.second.branchType() == edm::InEvent) {
        branchIDToWrapperIndex_[prod.second.branchID()] = index++;
      }
    }
  }
  for (auto& cache : cachedWrappers_) {
    cache.resize(nProdsInEvent);
  }
}

void RepeatingCachedRootSource::beginJob() {
  ProcessConfiguration processConfiguration;
  processConfiguration.setParameterSetID(ParameterSet::emptyParameterSetID());
  processConfiguration.setProcessConfigurationID();

  //Thinned collection associations are not supported at this time
  EventPrincipal eventPrincipal(productRegistry(),
                                branchIDListHelper(),
                                std::make_shared<ThinnedAssociationsHelper>(),
                                processConfiguration,
                                nullptr);
  {
    RunNumber_t run = 0;
    LuminosityBlockNumber_t lumi = 0;
    auto itAux = eventAuxs_.begin();
    auto itGetter = getters_.begin();
    for (auto& cache : cachedWrappers_) {
      rootFile_->nextEventEntry();
      rootFile_->readCurrentEvent(eventPrincipal);
      auto const& aux = eventPrincipal.aux();
      *(itAux++) = aux;
      if (0 == run) {
        run = aux.run();
        lumi = aux.luminosityBlock();
      } else {
        if (run != aux.run()) {
          throw cms::Exception("EventsWithDifferentRuns") << "The requested events to cache are from different Runs";
        }
        if (lumi != aux.luminosityBlock()) {
          throw cms::Exception("EventsWithDifferentLuminosityBlocks")
              << "The requested events to cache are from different LuminosityBlocks";
        }
      }
      selectionIDs_ = eventPrincipal.eventSelectionIDs();
      branchListIndexes_ = eventPrincipal.branchListIndexes();
      {
        auto reader = eventPrincipal.reader();
        auto& getter = *(itGetter++);
        for (auto const& branchToIndex : branchIDToWrapperIndex_) {
          cache[branchToIndex.second] = reader->getProduct(branchToIndex.first, &getter);
        }
      }
    }
    for (auto const& branchToIndex : branchIDToWrapperIndex_) {
      auto pid = eventPrincipal.branchIDToProductID(branchToIndex.first);
      productIDToWrapperIndex_[pid] = branchToIndex.second;
    }
    rootFile_->rewind();
  }
}

void RepeatingCachedRootSource::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.setComment(
      "Read only a few Events from one EDM/Root file, and repeat them in sequence. The Events are required to be from "
      "the same Run and LuminosityBlock.");
  desc.addUntracked<std::string>("fileName")->setComment("Name of file to be processed.");
  desc.addUntracked<unsigned int>("repeatNEvents", 10U)
      ->setComment("Number of events to read from file and then repeat in sequence.");
  desc.addUntracked<unsigned int>("skipEvents", 0);
  ProductSelectorRules::fillDescription(desc, "inputCommands");
  InputSource::fillDescription(desc);

  descriptions.add("source", desc);
}

//
// member functions
//

std::unique_ptr<RootFile> RepeatingCachedRootSource::makeRootFile(
    std::string const& logicalFileName,
    std::string const& pName,
    bool isSkipping,
    std::shared_ptr<InputFile> filePtr,
    std::shared_ptr<EventSkipperByID> skipper,
    std::shared_ptr<DuplicateChecker> duplicateChecker,
    std::vector<std::shared_ptr<IndexIntoFile>>& indexesIntoFiles) {
  return std::make_unique<RootFile>(pName,
                                    processConfiguration(),
                                    logicalFileName,
                                    filePtr,
                                    skipper,
                                    isSkipping,
                                    remainingEvents(),
                                    remainingLuminosityBlocks(),
                                    1,
                                    roottree::defaultCacheSize,  //treeCacheSize_,
                                    -1,                          //treeMaxVirtualSize(),
                                    processingMode(),
                                    runHelper_,
                                    false,  //noRunLumiSort_
                                    true,   //noEventSort_,
                                    selectorRules_,
                                    InputType::Primary,
                                    branchIDListHelper(),
                                    processBlockHelper().get(),
                                    thinnedAssociationsHelper(),
                                    nullptr,  // associationsFromSecondary
                                    duplicateChecker,
                                    false,  //dropDescendants(),
                                    processHistoryRegistryForUpdate(),
                                    indexesIntoFiles,
                                    0,  //currentIndexIntoFile,
                                    orderedProcessHistoryIDs_,
                                    false,   //bypassVersionCheck(),
                                    true,    //labelRawDataLikeMC(),
                                    false,   //usingGoToEvent_,
                                    true,    //enablePrefetching_,
                                    false);  //enforceGUIDInFileName_);
}

std::shared_ptr<WrapperBase> RepeatingCachedRootSource::getProduct(unsigned int iStreamIndex,
                                                                   BranchID const& k,
                                                                   EDProductGetter const* ep) const {
  return cachedWrappers_[streamToCacheIndex_[iStreamIndex]][branchIDToWrapperIndex_.find(k)->second];
}

RepeatingCachedRootSource::ItemType RepeatingCachedRootSource::getNextItemType() {
  auto v = presentState_;
  switch (presentState_) {
    case IsFile:
      presentState_ = IsRun;
      break;
    case IsRun:
      presentState_ = IsLumi;
      break;
    case IsLumi:
      presentState_ = IsEvent;
      break;
    default:
      break;
  }
  return v;
}

void RepeatingCachedRootSource::readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal) {
  rootFile_->readLuminosityBlock_(lumiPrincipal);
}

std::shared_ptr<LuminosityBlockAuxiliary> RepeatingCachedRootSource::readLuminosityBlockAuxiliary_() {
  return rootFile_->readLuminosityBlockAuxiliary_();
}
void RepeatingCachedRootSource::readEvent_(EventPrincipal& eventPrincipal) {
  auto index = eventIndex_++;

  auto repeatedIndex = index % cachedWrappers_.size();

  auto const& aux = eventAuxs_[repeatedIndex];

  auto history = processHistoryRegistry().getMapped(aux.processHistoryID());

  streamToCacheIndex_[eventPrincipal.streamID().value()] = repeatedIndex;
  eventPrincipal.fillEventPrincipal(aux,
                                    history,
                                    selectionIDs_,
                                    branchListIndexes_,
                                    EventToProcessBlockIndexes(),
                                    provRetriever_,
                                    &delayedReaders_[eventPrincipal.streamID().value()]);
}

std::shared_ptr<RunAuxiliary> RepeatingCachedRootSource::readRunAuxiliary_() {
  return rootFile_->readRunAuxiliary_();
  ;
}

void RepeatingCachedRootSource::readRun_(RunPrincipal& runPrincipal) { rootFile_->readRun_(runPrincipal); }

bool RepeatingCachedRootSource::readIt(EventID const& id,
                                       EventPrincipal& eventPrincipal,
                                       StreamContext& streamContext) {
  return false;
}

void RepeatingCachedRootSource::skip(int offset) {}

bool RepeatingCachedRootSource::goToEvent_(EventID const& eventID) { return false; }

void RepeatingCachedRootSource::fillProcessBlockHelper_() { rootFile_->fillProcessBlockHelper_(); }

bool RepeatingCachedRootSource::nextProcessBlock_(ProcessBlockPrincipal& processBlockPrincipal) {
  return rootFile_->nextProcessBlock_(processBlockPrincipal);
}

void RepeatingCachedRootSource::readProcessBlock_(ProcessBlockPrincipal& processBlockPrincipal) {
  rootFile_->readProcessBlock_(processBlockPrincipal);
}

WrapperBase const* RepeatingCachedRootSource::RCProductGetter::getIt(ProductID const& iPID) const {
  auto itFound = map_->find(iPID);
  if (itFound == map_->end()) {
    return nullptr;
  }
  return (*wrappers_)[itFound->second].get();
}

std::optional<std::tuple<WrapperBase const*, unsigned int>>
RepeatingCachedRootSource::RCProductGetter::getThinnedProduct(ProductID const&, unsigned int key) const {
  return {};
};

void RepeatingCachedRootSource::RCProductGetter::getThinnedProducts(ProductID const& pid,
                                                                    std::vector<WrapperBase const*>& foundContainers,
                                                                    std::vector<unsigned int>& keys) const {}

OptionalThinnedKey RepeatingCachedRootSource::RCProductGetter::getThinnedKeyFrom(ProductID const& parent,
                                                                                 unsigned int key,
                                                                                 ProductID const& thinned) const {
  return {};
}
unsigned int RepeatingCachedRootSource::RCProductGetter::transitionIndex_() const { return 0; }

//
// const member functions
//

//
// static member functions
//

DEFINE_FWK_INPUT_SOURCE(RepeatingCachedRootSource);
