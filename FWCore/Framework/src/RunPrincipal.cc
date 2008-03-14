#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/src/Group.h"

namespace edm {
  void
  RunPrincipal::addOrReplaceGroup(std::auto_ptr<Group> g) {

    Group* group = getExistingGroup(*g);
    if (group != 0) {

      assert(group->branchEntryDescription() != 0);
      if (!group->productUnavailable()) {
        assert(group->product() != 0);
      }
      assert(g->branchEntryDescription() != 0);
      if (!g->productUnavailable()) {
        assert(g->product() != 0);
      }

      group->mergeGroup(g.get());
    } else {
      addGroup_(g);
    }
  }

  void
  RunPrincipal::mergeRun(boost::shared_ptr<RunPrincipal> rp) {

    aux_.mergeAuxiliary(rp->aux());

    for (Principal::const_iterator i = rp->begin(), iEnd = rp->end(); i != iEnd; ++i) {

      std::auto_ptr<Group> g(new Group());
      g->swap(**i);

      addOrReplaceGroup(g);
    }
  }
}
