#ifndef DataFormats_Provenance_BranchChildren_h
#define DataFormats_Provenance_BranchChildren_h

/*----------------------------------------------------------------------
  
BranchChildren: Dependency information between branches.

----------------------------------------------------------------------*/

#include <map>
#include <set>
#include "DataFormats/Provenance/interface/BranchID.h"

namespace edm {
  typedef std::map<BranchID, std::set<BranchID> > BranchChildren;
}
#endif
