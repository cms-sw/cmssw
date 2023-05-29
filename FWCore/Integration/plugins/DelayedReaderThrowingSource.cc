#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Sources/interface/IDGeneratorSourceBase.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Utilities/interface/GetPassID.h"

namespace edm {
  namespace {
    class ThrowingDelayedReader : public DelayedReader {
    public:
      ThrowingDelayedReader(
          signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadSource,
          signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadSource)
          : preEventReadFromSourceSignal_(preEventReadSource),
            postEventReadFromSourceSignal_(postEventReadSource),
            e_(std::make_exception_ptr(cms::Exception("TEST"))) {}

      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadFromSourceSignal()
          const final {
        return preEventReadFromSourceSignal_;
      }
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadFromSourceSignal()
          const final {
        return postEventReadFromSourceSignal_;
      }

    private:
      std::shared_ptr<WrapperBase> getProduct_(BranchID const& k, EDProductGetter const* ep) final {
        try {
          std::rethrow_exception(e_);
        } catch (cms::Exception const& iE) {
          //avoid adding to the context for each call
          auto copyException = iE;
          copyException.addContext("called ThrowingDelayedReader");
          throw copyException;
        }
      }
      void mergeReaders_(DelayedReader*) final{};
      void reset_() final{};

      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadFromSourceSignal_ =
          nullptr;
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadFromSourceSignal_ =
          nullptr;
      std::exception_ptr e_;
    };
  }  // namespace

  class DelayedReaderThrowingSource : public IDGeneratorSourceBase<InputSource> {
  public:
    explicit DelayedReaderThrowingSource(ParameterSet const&, InputSourceDescription const&);
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    bool setRunAndEventInfo(EventID& id, TimeValue_t& time, edm::EventAuxiliary::ExperimentType&) override;
    void readEvent_(edm::EventPrincipal&) override;

    std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> resourceSharedWithDelayedReader_() override {
      return std::make_pair(resourceSharedWithDelayedReaderPtr_.get(), mutexSharedWithDelayedReader_.get());
    }

    ThrowingDelayedReader delayedReader_;
    ProcessHistoryID historyID_;

    std::unique_ptr<SharedResourcesAcquirer>
        resourceSharedWithDelayedReaderPtr_;  // We do not use propagate_const because the acquirer is itself mutable.
    std::shared_ptr<std::recursive_mutex> mutexSharedWithDelayedReader_;
  };

  DelayedReaderThrowingSource::DelayedReaderThrowingSource(ParameterSet const& pset, InputSourceDescription const& desc)
      : IDGeneratorSourceBase<InputSource>(pset, desc, false),
        delayedReader_(&preEventReadFromSourceSignal_, &postEventReadFromSourceSignal_) {
    auto resources = SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader();
    resourceSharedWithDelayedReaderPtr_ = std::make_unique<SharedResourcesAcquirer>(std::move(resources.first));
    mutexSharedWithDelayedReader_ = resources.second;

    ParameterSet dummy;
    dummy.registerIt();
    auto twd = TypeWithDict::byTypeInfo(typeid(edmtest::IntProduct));

    std::vector<BranchDescription> branches;
    for (auto const& label : pset.getUntrackedParameter<std::vector<std::string>>("labels")) {
      branches.push_back(BranchDescription(InEvent,
                                           label,        //module label
                                           "INPUTTEST",  //can't be the present process name
                                           twd.userClassName(),
                                           twd.friendlyClassName(),
                                           "",  //product instance name
                                           "",  //module name which isn't set for items not produced
                                           dummy.id(),
                                           twd,
                                           false  //not produced
                                           ));
      branches.back().setOnDemand(true);  //says we use delayed reader
    }
    productRegistryUpdate().updateFromInput(branches);

    ProcessHistory ph;
    ph.emplace_back("INPUTTEST", dummy.id(), PROJECT_VERSION, getPassID());
    processHistoryRegistry().registerProcessHistory(ph);
    historyID_ = ph.id();

    BranchIDLists bilists(1);
    for (auto const& branch : branches) {
      bilists[0].emplace_back(branch.branchID().id());
    }
    branchIDListHelper()->updateFromInput(bilists);
  }

  bool DelayedReaderThrowingSource::setRunAndEventInfo(EventID&, TimeValue_t&, edm::EventAuxiliary::ExperimentType&) {
    return true;
  }

  void DelayedReaderThrowingSource::readEvent_(edm::EventPrincipal& e) {
    BranchListIndexes indexes(1, static_cast<unsigned short>(0));
    branchIDListHelper()->fixBranchListIndexes(indexes);
    doReadEventWithDelayedReader(e, historyID_, EventSelectionIDVector(), std::move(indexes), &delayedReader_);
  }

  void DelayedReaderThrowingSource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Throws an exception when the DelayedReader is used.");
    IDGeneratorSourceBase<InputSource>::fillDescription(desc);
    desc.addUntracked<std::vector<std::string>>("labels", {{"test"}});
    descriptions.add("source", desc);
  }
}  // namespace edm

using edm::DelayedReaderThrowingSource;
DEFINE_FWK_INPUT_SOURCE(DelayedReaderThrowingSource);
