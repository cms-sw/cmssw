#ifndef DataFormats_Provenance_BranchChildren_h
#define DataFormats_Provenance_BranchChildren_h

/*----------------------------------------------------------------------
  
BranchChildren: Dependency information between branches.

----------------------------------------------------------------------*/
#if (not defined __INCLUDE_LEVEL__ or __INCLUDE_LEVEL__ > 0) and \
    not defined(DataFormats_Provenance_ProductDependencies_h)
#error The name BranchChildren is deprecated, please use ProductDependencies instead.
#endif

#include <map>
#include <set>
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductDescriptionFwd.h"

namespace edm {

  class BranchChildren {
  private:
    typedef std::set<BranchID> BranchIDSet;
    typedef std::map<BranchID, BranchIDSet> map_t;

  public:
    // Clear all information.
    void clear();

    // Insert a parent with no children.
    void insertEmpty(BranchID parent);

    // Insert a new child for the given parent.
    void insertChild(BranchID parent, BranchID child);

    // Look up all the descendants of the given parent, and insert them
    // into descendants. N.B.: this does not clear out descendants first;
    // it only appends *new* elements to the collection.
    void appendToDescendants(ProductDescription const& parent,
                             BranchIDSet& descendants,
                             std::map<BranchID, BranchID> const& droppedToKeptAlias) const;

    // const accessor for the data
    map_t const& childLookup() const { return childLookup_; }

    map_t& mutableChildLookup() { return childLookup_; }

  private:
    map_t childLookup_;

    void append_(map_t const& lookup,
                 BranchID item,
                 BranchIDSet& itemSet,
                 std::map<BranchID, BranchID> const& droppedToKeptAlias) const;
  };

}  // namespace edm
#endif
