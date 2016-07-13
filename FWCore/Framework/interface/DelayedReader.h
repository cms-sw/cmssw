#ifndef FWCore_Framework_DelayedReader_h
#define FWCore_Framework_DelayedReader_h

/*----------------------------------------------------------------------

DelayedReader: The abstract interface through which the Principal
uses input sources to retrieve EDProducts from external storage.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/WrapperBase.h"

#include <memory>

namespace edm {

  class BranchKey;
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
    std::unique_ptr<WrapperBase> getProduct(BranchKey const& k,
                                            EDProductGetter const* ep,
                                            signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preReadFromSourceSignal = nullptr,
                                            signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postReadFromSourceSignal = nullptr,
                                            ModuleCallingContext const* mcc = nullptr);

    void mergeReaders(DelayedReader* other) {mergeReaders_(other);}
    void reset() {reset_();}
    
    SharedResourcesAcquirer* sharedResources() const {
      return sharedResources_();
    }
    
    
    
  private:
    virtual std::unique_ptr<WrapperBase> getProduct_(BranchKey const& k, EDProductGetter const* ep) = 0;
    virtual void mergeReaders_(DelayedReader*) = 0;
    virtual void reset_() = 0;
    virtual SharedResourcesAcquirer* sharedResources_() const;
  };
}

#endif
