#ifndef FWCore_Framework_DelayedReader_h
#define FWCore_Framework_DelayedReader_h

/*----------------------------------------------------------------------

DelayedReader: The abstract interface through which the Principal
uses input sources to retrieve EDProducts from external storage.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/WrapperBase.h"

#include <memory>
#include <mutex>

namespace edm {

  class BranchID;
  class EDProductGetter;
  class ModuleCallingContext;
  class SharedResourcesAcquirer;
  class StreamContext;

  namespace signalslot {
    template <typename T> class Signal;
  }

  class DelayedReader {
  public:
    virtual ~DelayedReader();
    std::unique_ptr<WrapperBase> getProduct(BranchID const& k,
                                            EDProductGetter const* ep,
                                            ModuleCallingContext const* mcc = nullptr);

    void mergeReaders(DelayedReader* other) {mergeReaders_(other);}
    void reset() {reset_();}
    
    std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> sharedResources() const {
      return sharedResources_();
    }
    

    virtual signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadFromSourceSignal() const = 0;
    virtual signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadFromSourceSignal() const = 0;

    
  private:
    virtual std::unique_ptr<WrapperBase> getProduct_(BranchID const& k, EDProductGetter const* ep) = 0;
    virtual void mergeReaders_(DelayedReader*) = 0;
    virtual void reset_() = 0;
    virtual std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> sharedResources_() const;

  };
}

#endif
