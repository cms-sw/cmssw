#ifndef FWCore_Framework_DelayedReader_h
#define FWCore_Framework_DelayedReader_h

/*----------------------------------------------------------------------

DelayedReader: The abstract interface through which the Principal
uses input sources to retrieve EDProducts from external storage.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/WrapperOwningHolder.h"

#include <memory>

namespace edm {
  class BranchKey;
  class EDProductGetter;
  class WrapperInterfaceBase;
  class SharedResourcesAcquirer;
  class DelayedReader {
  public:
    virtual ~DelayedReader();
    WrapperOwningHolder getProduct(BranchKey const& k, WrapperInterfaceBase const* interface, EDProductGetter const* ep);
    
    void mergeReaders(DelayedReader* other) {mergeReaders_(other);}
    void reset() {reset_();}
    
    SharedResourcesAcquirer* sharedResources() const {
      return sharedResources_();
    }
    
    
    
  private:
    virtual WrapperOwningHolder getProduct_(BranchKey const& k,
                                            WrapperInterfaceBase const* interface,
                                            EDProductGetter const* ep) const = 0;
    virtual void mergeReaders_(DelayedReader*) = 0;
    virtual void reset_() = 0;
    virtual SharedResourcesAcquirer* sharedResources_() const;
  };
}

#endif
