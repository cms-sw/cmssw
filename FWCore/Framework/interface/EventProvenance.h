#ifndef Framework_EventProvenance_h
#define Framework_EventProvenance_h

/*----------------------------------------------------------------------
  
EventProvenance: The Provenance of all the products in an event.

$Id: EventProvenance.h,v 1.2 2005/07/14 22:50:52 wmtan Exp $
----------------------------------------------------------------------*/
#include <vector>

#include "FWCore/Framework/interface/Provenance.h"
namespace edm {

  struct EventProvenance {
    std::vector<Provenance> data_;  // One entry per EDProduct
  };
}
#endif
