#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/src/Group.h"

namespace edm {
  void
  RunPrincipal::addOrReplaceGroup(std::auto_ptr<Group> g) {
    Group const* group = getExistingGroup(*g);
    if (group != 0) {
      replaceGroup(g);
    } else {
      addGroup_(g);
    }
  }
}
