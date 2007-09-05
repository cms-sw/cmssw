#ifndef FWCore_Framework_DelayedReader_h
#define FWCore_Framework_DelayedReader_h

/*----------------------------------------------------------------------
  
DelayedReader: The abstract interface through which the EventPrincipal
uses input sources to retrieve EDProducts from external storage.

$Id: DelayedReader.h,v 1.7 2007/08/15 22:45:41 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include "DataFormats/Provenance/interface/ProvenanceDelayedReader.h"

namespace edm {
  class BranchKey;
  class EDProduct;
  class EDProductGetter;
  class DelayedReader : public ProvenanceDelayedReader {
  public:
    virtual ~DelayedReader();
    virtual std::auto_ptr<EDProduct> getProduct(BranchKey const& k, EDProductGetter const* ep) const = 0;
  };
}

#endif
