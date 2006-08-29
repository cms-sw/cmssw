#ifndef Common_EventProvenance_h
#define Common_EventProvenance_h

/*----------------------------------------------------------------------
  
EventProvenance: The Event dependent Provenance of all the products in an event.

$Id: EventProvenance.h,v 1.1 2006/02/08 00:44:23 wmtan Exp $
----------------------------------------------------------------------*/
#include <vector>

#include "DataFormats/Common/interface/BranchEntryDescription.h"
namespace edm {

  struct EventProvenance {
    EventProvenance() : data_() {}
    std::vector<BranchEntryDescription> data_;  // One entry per EDProduct
  };
}
#endif
