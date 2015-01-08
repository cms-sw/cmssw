#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"

namespace edm {

  ConsumesInfo::ConsumesInfo(TypeID const& iType,
                             char const* iLabel,
                             char const* iInstance,
                             char const* iProcess,
                             BranchType iBranchType,
                             KindOfType iKindOfType,
                             bool iAlwaysGets,
                             bool iSkipCurrentProcess_) :
    type_(iType),
    label_(iLabel),
    instance_(iInstance),
    process_(iProcess),
    branchType_(iBranchType),
    kindOfType_(iKindOfType),
    alwaysGets_(iAlwaysGets),
    skipCurrentProcess_(iSkipCurrentProcess_) {
  }
}
