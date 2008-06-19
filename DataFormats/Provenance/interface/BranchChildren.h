#ifndef DataFormats_Provenance_BranchChildren_h
#define DataFormats_Provenance_BranchChildren_h

/*----------------------------------------------------------------------
  
BranchChildren: Dependency information between branches.

----------------------------------------------------------------------*/

#include <map>
#include <set>
#include "DataFormats/Provenance/interface/BranchID.h"

namespace edm {

  class BranchChildren
  {
  public:

    void clear();
    void insertEmpty(BranchID parent);
    void insertChild(BranchID parent, BranchID child);

  private:
    typedef std::map<BranchID, std::set<BranchID> > map_t;
    map_t childLookup_;
    map_t parentLookup_;

    void fillParentLookup_();
  };
  //typedef std::map<BranchID, std::set<BranchID> > BranchChildren;
}
#endif
