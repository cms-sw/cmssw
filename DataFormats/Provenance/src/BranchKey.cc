/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/
#include <ostream>

#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"


namespace edm {
  BranchKey::BranchKey(BranchDescription const& desc) :
    friendlyClassName_(desc.friendlyClassName()),
    moduleLabel_(desc.moduleLabel()),
    productInstanceName_(desc.productInstanceName()),
    processName_(desc.processName()) {}

  BranchKey::BranchKey(ConstBranchDescription const& desc) :
    friendlyClassName_(desc.friendlyClassName()),
    moduleLabel_(desc.moduleLabel()),
    productInstanceName_(desc.productInstanceName()),
    processName_(desc.processName()) {}

  std::ostream&
  operator<<(std::ostream& os, BranchKey const& bk) {
    os << "BranchKey("
       << bk.friendlyClassName() << ", "
       << bk.moduleLabel() << ", "
       << bk.productInstanceName() << ", "
       << bk.processName() << ')';
    return os;
  }
}
