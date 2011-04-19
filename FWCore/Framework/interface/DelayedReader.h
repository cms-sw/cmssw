#ifndef FWCore_Framework_DelayedReader_h
#define FWCore_Framework_DelayedReader_h

/*----------------------------------------------------------------------

DelayedReader: The abstract interface through which the EventPrincipal
uses input sources to retrieve EDProducts from external storage.

----------------------------------------------------------------------*/

#include <memory>
#include "boost/shared_ptr.hpp"
#include "DataFormats/Common/interface/WrapperHolder.h"

namespace edm {
  struct BranchKey;
  class EDProductGetter;
  class WrapperInterfaceBase;
  class DelayedReader {
  public:
    virtual ~DelayedReader();
    WrapperHolder getProduct(BranchKey const& k, WrapperInterfaceBase const* interface, EDProductGetter const* ep) {
      return getProduct_(k, interface, ep);
    }
    void mergeReaders(boost::shared_ptr<DelayedReader> other) {mergeReaders_(other);}
  private:
    virtual WrapperHolder getProduct_(BranchKey const& k, WrapperInterfaceBase const* interface, EDProductGetter const* ep) const = 0;
    virtual void mergeReaders_(boost::shared_ptr<DelayedReader>) {}
  };
}

#endif
