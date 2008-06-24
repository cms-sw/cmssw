#include "DataFormats/Provenance/interface/BranchChildren.h"

namespace edm {
  void
  BranchChildren::append_(map_t const& lookup, BranchID item, BranchIDSet& itemSet) const {
    BranchIDSet const& items = const_cast<map_t &>(lookup)[item];
    // For each parent(child)
    for (BranchIDSet::const_iterator ci = items.begin(), ce = items.end();
	ci != ce; ++ci) {
      // Insert the BranchID of the parents(children) into the set of ancestors(descendents).
      // If the insert succeeds, append recursively.
      if (itemSet.insert(*ci).second) {
	append_(lookup, *ci, itemSet);
      }
    }
  }

  void
  BranchChildren::clear() {
    childLookup_.clear();
    parentLookup_.clear();
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
  BranchChildren::appendToAncestors(BranchID child, BranchIDSet& ancestors) const {
    fillParentLookupIfNecessary_();
    ancestors.insert(child);
    append_(parentLookup_, child, ancestors);
  }

  void
  BranchChildren::appendToDescendents(BranchID parent, BranchIDSet& descendents) const {
    descendents.insert(parent);
    append_(childLookup_, parent, descendents);
  }

  void
  BranchChildren::fillParentLookupIfNecessary_() const {
    if (parentLookup_.empty()) {
      // for each parent ...
      for (map_t::const_iterator parent = childLookup_.begin(), e = childLookup_.end(); parent != e; ++parent) {
	// for each child of that parent
	for (BranchIDSet::const_iterator ci = parent->second.begin(), ce = parent->second.end();
		 ci != ce; ++ci) {
		// insert the BranchID of the parent into the set of BranchIDs for this child.
	  parentLookup_[*ci].insert(parent->first);
	}	    
      }
    }
  }
}
