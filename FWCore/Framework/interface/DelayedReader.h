#ifndef FWCore_Framework_DelayedReader_h
#define FWCore_Framework_DelayedReader_h

/*----------------------------------------------------------------------
  
DelayedReader: The abstract interface through which the EventPrincipal
uses input sources to retrieve EDProducts from external storage.

$Id: DelayedReader.h,v 1.8 2007/09/05 23:50:39 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include "boost/shared_ptr.hpp"
#include "DataFormats/Provenance/interface/ProvenanceDelayedReader.h"
#include "DataFormats/Common/interface/EDProduct.h"

namespace edm {
  class BranchKey;
  class EDProduct;
  class EDProductGetter;
  class DelayedReader : public ProvenanceDelayedReader {
  public:
    virtual ~DelayedReader();
    std::auto_ptr<EDProduct> getProduct(BranchKey const& k, EDProductGetter const* ep) {
      return getProduct_(k, ep);
    }
    void mergeReaders(boost::shared_ptr<DelayedReader> other) {mergeReaders_(other);}
  private:
    virtual std::auto_ptr<EDProduct> getProduct_(BranchKey const& k, EDProductGetter const* ep) const = 0;
    virtual void mergeReaders_(boost::shared_ptr<DelayedReader>) {}
  };
}

#endif
