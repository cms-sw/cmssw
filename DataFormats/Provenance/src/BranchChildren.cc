#include "DataFormats/Provenance/interface/BranchChildren.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"

namespace edm {
  void BranchChildren::append_(map_t const& lookup,
                               BranchID item,
                               BranchIDSet& itemSet,
                               std::map<BranchID, BranchID> const& droppedToKeptAlias) const {
    auto const iter = lookup.find(item);
    if (iter != lookup.end()) {
      BranchIDSet const& branchIDs = iter->second;
      for (BranchID const& branchID : branchIDs) {
        auto it = droppedToKeptAlias.find(branchID);
        // Insert the BranchID of the children into the set of descendants.
        // If the insert succeeds, append recursively.
        if (it == droppedToKeptAlias.end()) {
          // Normal case. Not an EDAlias.
          if (itemSet.insert(branchID).second) {
            append_(lookup, branchID, itemSet, droppedToKeptAlias);
          }
        } else {
          // In this case, we want to put the EDAlias in the
          // set of things to drop because that is what really
          // needs to be dropped, but the recursive search in
          // the lookup map must continue with the original BranchID
          // because that is what the lookup map contains.
          if (itemSet.insert(it->second).second) {
            append_(lookup, branchID, itemSet, droppedToKeptAlias);
          }
        }
      }
    }
  }

  void BranchChildren::clear() { childLookup_.clear(); }

  void BranchChildren::insertEmpty(BranchID parent) { childLookup_.insert(std::make_pair(parent, BranchIDSet())); }

  void BranchChildren::insertChild(BranchID parent, BranchID child) { childLookup_[parent].insert(child); }

  void BranchChildren::appendToDescendants(BranchDescription const& parent,
                                           BranchIDSet& descendants,
                                           std::map<BranchID, BranchID> const& droppedToKeptAlias) const {
    descendants.insert(parent.branchID());
    // A little tricky here. The child lookup map is filled with the
    // BranchID of the original product even if there was an EDAlias
    // and the EDAlias was saved and the original branch dropped.
    append_(childLookup_, parent.originalBranchID(), descendants, droppedToKeptAlias);
  }
}  // namespace edm
