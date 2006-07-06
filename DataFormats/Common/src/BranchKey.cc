/*----------------------------------------------------------------------
  
$Id: BranchKey.cc,v 1.2.2.1 2006/07/01 06:21:58 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>

#include "DataFormats/Common/interface/BranchKey.h"
#include "DataFormats/Common/interface/BranchDescription.h"


namespace edm
{
  BranchKey::BranchKey(BranchDescription const& desc) :
    friendlyClassName_(desc.friendlyClassName()),
    moduleLabel_(desc.moduleLabel()),
    productInstanceName_(desc.productInstanceName()),
    processName_(desc.processName()) {}

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
