/*----------------------------------------------------------------------
  
$Id: BranchKey.cc,v 1.5 2005/07/30 23:47:52 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>

#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/BranchDescription.h"


namespace edm
{
  BranchKey::BranchKey(BranchDescription const& desc) :
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
