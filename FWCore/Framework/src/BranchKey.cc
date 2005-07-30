/*----------------------------------------------------------------------
  
$Id: BranchKey.cc,v 1.4 2005/07/30 04:44:03 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>

#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/ProductDescription.h"


namespace edm
{
  BranchKey::BranchKey(ProductDescription const& desc) :
    friendlyClassName_(desc.friendlyClassName_),
    moduleLabel_(desc.module.moduleLabel_),
    productInstanceName_(desc.productInstanceName_),
    processName_(desc.module.processName_) {}

  std::ostream&
  operator<<(std::ostream& os, const BranchKey& bk) {
    os << "BranchKey("
       << bk.friendlyClassName_ << ", "
       << bk.moduleLabel_ << ", "
       << bk.productInstanceName_ << ", "
       << bk.processName_ << ')';
    return os;
  }

  
}
