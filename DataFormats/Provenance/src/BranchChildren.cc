#include "DataFormats/Provenance/interface/BranchChildren.h"

namespace edm {
  void
  BranchChildren::append_(map_t const& lookup, BranchID item, BranchIDSet& itemSet) const {
    BranchIDSet const& items = const_cast<map_t &>(lookup)[item];
    // For each parent(child)
    for (BranchIDSet::const_iterator ci = items.begin(), ce = items.end();
	ci != ce; ++ci) {
      // Insert the BranchID of the parents(children) into the set of ancestors(descendants).
      // If the insert succeeds, append recursively.
      if (itemSet.insert(*ci).second) {
	append_(lookup, *ci, itemSet);
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
