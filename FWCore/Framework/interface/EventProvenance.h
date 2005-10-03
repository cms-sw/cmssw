#ifndef Framework_EventProvenance_h
#define Framework_EventProvenance_h

/*----------------------------------------------------------------------
  
EventProvenance: The Event dependent Provenance of all the products in an event.

$Id: EventProvenance.h,v 1.3 2005/09/01 05:36:45 wmtan Exp $
----------------------------------------------------------------------*/
#include <vector>

#include "FWCore/Framework/interface/BranchEntryDescription.h"
namespace edm {

  struct EventProvenance {
    std::vector<BranchEntryDescription> data_;  // One entry per EDProduct
  };
}
#endif
