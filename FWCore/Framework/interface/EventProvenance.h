#ifndef EDM_EVENTPROVENANCE_H
#define EDM_EVENTPROVENANCE_H

/*----------------------------------------------------------------------
  
EventProvenance: The Provenance of all the products in an event.

$Id: EventProvenance.h,v 1.1 2005/04/04 23:26:21 wmtan Exp $
----------------------------------------------------------------------*/
#include <vector>

#include "FWCore/CoreFramework/interface/Provenance.h"
namespace edm {

  struct EventProvenance {
    std::vector<Provenance> data_;  // One entry per EDProduct
  };
}
#endif
