#include "DataFormats/Provenance/interface/BranchChildren.h"

namespace edm {
  void
  BranchChildren::append_(map_t const& lookup, BranchID item, BranchIDSet& itemSet) const {
    auto const iter = lookup.find(item);
    if(iter != lookup.end()) {
      BranchIDSet const& branchIDs = iter->second;
      for(BranchID const& branchID : branchIDs) {
        // Insert the BranchID of the parents(children) into the set of ancestors(descendants).
        // If the insert succeeds, append recursively.
        if(itemSet.insert(branchID).second) {
          append_(lookup, branchID, itemSet);
        }
      }
    }
  }

  void
  BranchChildren::clear() {
    childLookup_.clear();
  }

  void
  BranchChildren::insertEmpty(BranchID parent) {
    childLookup_.insert(std::make_pair(parent, BranchIDSet()));
  }

  void
  BranchChildren::insertChild(BranchID parent, BranchID child) {
    childLookup_[parent].insert(child);
  }

  void
  BranchChildren::appendToDescendants(BranchID parent, BranchIDSet& descendants) const {
    descendants.insert(parent);
    append_(childLookup_, parent, descendants);
  }
}
