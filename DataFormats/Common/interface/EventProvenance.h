#ifndef Common_EventProvenance_h
#define Common_EventProvenance_h

/*----------------------------------------------------------------------
  
EventProvenance: The Event dependent Provenance of all the products in an event.

$Id: EventProvenance.h,v 1.4 2005/10/03 19:54:34 wmtan Exp $
----------------------------------------------------------------------*/
#include <vector>

#include "DataFormats/Common/interface/BranchEntryDescription.h"
namespace edm {

  struct EventProvenance {
    std::vector<BranchEntryDescription> data_;  // One entry per EDProduct
  };
}
#endif
