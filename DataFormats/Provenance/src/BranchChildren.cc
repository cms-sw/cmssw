#include "DataFormats/Provenance/interface/BranchChildren.h"

namespace edm
{
  void
  BranchChildren::clear()
  {
    childLookup_.clear();
    parentLookup_.clear();
  }

  void
  BranchChildren::insertEmpty(BranchID parent)
  {
    childLookup_.insert(std::make_pair(parent, std::set<BranchID>()));
  }

  void
  BranchChildren::insertChild(BranchID parent, BranchID child)
  {
    childLookup_[parent].insert(child);
  }

  void
  BranchChildren::fillParentLookup_()
  {
    if (parentLookup_.empty())
      {
	// nothing yet.
      }
  }
}
