#include "FWCore/Framework/interface/BranchEntryDescription.h"

/*----------------------------------------------------------------------

$Id: BranchEntryDescription.cc,v 1.2 2005/09/12 23:32:13 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  BranchEntryDescription::BranchEntryDescription() :
    productID_(),
    parents(),
    cid(),
    status(Success)
  { }

  void
  BranchEntryDescription::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    os << "BranchEntryDescription for: " << cid;
  }

    
  bool
  BranchEntryDescription::operator==(BranchEntryDescription const& b) const {
    return
      productID_ == b.productID_
      && cid == b.cid
      && status == b.status
      && parents == b.parents;
  }

}
