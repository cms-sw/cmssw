/*----------------------------------------------------------------------
  
$Id: BranchKey.cc,v 1.6 2005/10/03 19:06:06 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>

#include "DataFormats/Common/interface/BranchKey.h"
#include "DataFormats/Common/interface/BranchDescription.h"


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
