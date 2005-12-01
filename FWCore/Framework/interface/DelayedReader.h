#ifndef Framework_DelayedReader_h
#define Framework_DelayedReader_h

/*----------------------------------------------------------------------
  
DelayedReader: The abstract interface through which the EventPrincipal
uses input sources to retrieve EDProducts from external storage.

$Id: DelayedReader.h,v 1.1 2005/09/28 05:26:37 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/EDProduct/interface/EDProduct.h"


namespace edm {
  class BranchKey;
  class EventPrincipal;
  class DelayedReader {
  public:
    virtual ~DelayedReader();

    virtual std::auto_ptr<EDProduct> get(BranchKey const& k, EventPrincipal const* ep) const = 0;
  };
}

#endif
