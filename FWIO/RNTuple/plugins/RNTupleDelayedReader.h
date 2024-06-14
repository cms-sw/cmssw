#if !defined(RNTupleDelayedReader_h)
#define RNTupleDelayedReader_h

#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace edm::input {
  class DataProductsRNTuple;

  class RNTupleDelayedReader : public DelayedReader {
  public:
    RNTupleDelayedReader(DataProductsRNTuple* iRNTuple, SharedResourcesAcquirer*, std::recursive_mutex*);

    void setEntry(int iEntry) { entry_ = iEntry; }

    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadFromSourceSignal()
        const final {
      return preEventReadFromSourceSignal_;
    }
    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadFromSourceSignal()
        const final {
      return postEventReadFromSourceSignal_;
    }

    void setSignals(
        signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadSource,
        signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadSource) {
      preEventReadFromSourceSignal_ = preEventReadSource;
      postEventReadFromSourceSignal_ = postEventReadSource;
    }

  private:
    std::shared_ptr<WrapperBase> getProduct_(BranchID const& k, EDProductGetter const* ep) final;
    void mergeReaders_(DelayedReader*) final {}
    void reset_() final {}
    std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> sharedResources_() const final;

    DataProductsRNTuple* rntuple_;
    SharedResourcesAcquirer* resourceAcquirer_;  // We do not use propagate_const because the acquirer is itself mutable.
    std::recursive_mutex* mutex_;
    InputType inputType_;
    int entry_ = 0;

    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadFromSourceSignal_ =
        nullptr;
    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadFromSourceSignal_ =
        nullptr;

    //If a fatal exception happens we need to make a copy so we can
    // rethrow that exception on other threads. This avoids RNTuple
    // non-exception safety problems on later calls to RNTuple.
    //All uses of the ROOT file are serialized
    CMS_SA_ALLOW mutable std::exception_ptr lastException_;
  };
}  // namespace edm::input

#endif
