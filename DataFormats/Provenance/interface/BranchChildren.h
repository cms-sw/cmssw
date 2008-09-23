#ifndef DataFormats_Provenance_BranchChildren_h
#define DataFormats_Provenance_BranchChildren_h

/*----------------------------------------------------------------------
  
BranchChildren: Dependency information between branches.

----------------------------------------------------------------------*/

#include <map>
#include <set>
#include "DataFormats/Provenance/interface/BranchID.h"

namespace edm {

  class BranchChildren {
  private:
    typedef std::set<BranchID> BranchIDSet;
  public:

    // Clear all information.
    void clear();

    // Insert a parent with no children.
    void insertEmpty(BranchID parent);

    // Insert a new child for the given parent.
    void insertChild(BranchID parent, BranchID child);

    // Look up all the descendents of the given parent, and insert them
    // into descendents. N.B.: this does not clear out descendents first;
    // it only appends *new* elements to the collection.
    void appendToDescendents(BranchID parent, BranchIDSet& descendents) const;

  private:
    typedef std::map<BranchID, BranchIDSet> map_t;
    map_t childLookup_;

    void append_(map_t const& lookup, BranchID item, BranchIDSet& itemSet) const;
  };

}
#endif
