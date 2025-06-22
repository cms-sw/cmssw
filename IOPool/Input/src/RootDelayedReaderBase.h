#ifndef IOPool_Input_RootDelayedReaderBase_h
#define IOPool_Input_RootDelayedReaderBase_h

/*----------------------------------------------------------------------

RootDelayedReaderBase.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "RootTree.h"

#include <map>
#include <memory>
#include <string>
#include <exception>

class TClass;
namespace edm {
  class InputFile;
  class RootTree;
  class SharedResourcesAcquirer;
  class Exception;

  //------------------------------------------------------------
  // Class RootDelayedReaderBase: pretends to support file reading.
  //

  class RootDelayedReaderBase : public DelayedReader {
  public:
    RootDelayedReaderBase() = default;

    ~RootDelayedReaderBase() override = default;

    RootDelayedReaderBase(RootDelayedReaderBase const&) = delete;             // Disallow copying and moving
    RootDelayedReaderBase& operator=(RootDelayedReaderBase const&) = delete;  // Disallow copying and moving

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

    virtual void readAllProductsNow(EDProductGetter const* ep) = 0;

  private:
    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadFromSourceSignal_ =
        nullptr;
    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadFromSourceSignal_ =
        nullptr;
  };  // class RootDelayedReaderBase
  //------------------------------------------------------------
}  // namespace edm
#endif
