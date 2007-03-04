/*----------------------------------------------------------------------
  
$Id: BranchKey.cc,v 1.3 2006/07/06 18:34:06 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>

#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"


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
