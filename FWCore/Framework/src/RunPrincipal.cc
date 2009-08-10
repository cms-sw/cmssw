#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Group.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

namespace edm {
  RunPrincipal::RunPrincipal(RunAuxiliary const& aux,
    boost::shared_ptr<ProductRegistry const> reg,
    ProcessConfiguration const& pc,
    boost::shared_ptr<BranchMapper> mapper,
    boost::shared_ptr<DelayedReader> rtrv) :
      Base(reg, pc, InRun, aux.processHistoryID_, mapper, rtrv),
      aux_(aux) {
    if (reg->productProduced(InRun)) {
      addToProcessHistory();
    }
    mapper->processHistoryID() = processHistoryID();
  }

  void
  RunPrincipal::addOrReplaceGroup(std::auto_ptr<Group> g) {

    Group* group = getExistingGroup(*g);
    if (group == 0) {
      addGroup_(g);
    } else if (group->productUnavailable() && group->branchDescription().produced()) {
      // In this case, group is for the current process, and the existing group is just
      // a placeholder. Just replace the existing group.
      assert(g->branchDescription().produced());
      replaceGroup(*g);
    } else {
      if (!group->productUnavailable()) {
        assert(group->product() != 0);
      }
      if (!g->productUnavailable()) {
        assert(g->product() != 0);
      }
      if(static_cast<bool> (g.get())) {
         //PrincipalCache holds onto the 'newest' version of a RunPrincipal for a given run
         // but our behavior is to keep the 'old' group and merge in the new one because if there
         // is no way to merge we keep the 'old' group
         edm::swap(*group,*g);
      }
      group->mergeGroup(g.get());
    }
  }

  void
  RunPrincipal::addGroupScheduled(ConstBranchDescription const& bd) {
    std::auto_ptr<Group> g(new Group(bd, ProductID(), productstatus::producerNotRun()));
    addGroupOrNoThrow(g);
  }

  void
  RunPrincipal::addGroupSource(ConstBranchDescription const& bd) {
    std::auto_ptr<Group> g(new Group(bd, ProductID(), productstatus::producerDidNotPutProduct()));
    addGroupOrNoThrow(g);
  }

  void
  RunPrincipal::addGroupIfNeeded(ConstBranchDescription const& bd) {
    if (getExistingGroup(bd.branchID()) == 0) {
      addGroup(bd, true);
    }
  }

  void
  RunPrincipal::addGroup(ConstBranchDescription const& bd, bool dropped) {
    std::auto_ptr<Group> g(new Group(bd, ProductID(), dropped));
    addOrReplaceGroup(g);
  }

  void
  RunPrincipal::addToGroup(boost::shared_ptr<EDProduct> prod,
	ConstBranchDescription const& bd,
	std::auto_ptr<ProductProvenance> productProvenance) {
    std::auto_ptr<Group> g(new Group(prod, bd, ProductID(), productProvenance));
    addOrReplaceGroup(g);
  }

  void 
  RunPrincipal::put(boost::shared_ptr<EDProduct> edp,
		ConstBranchDescription const& bd,
		std::auto_ptr<ProductProvenance> productProvenance) {

    if (edp.get() == 0) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Pointer")
	<< "put: Cannot put because auto_ptr to product is null."
	<< "\n";
    }
    branchMapperPtr()->insert(*productProvenance);
    // Group assumes ownership
    this->addToGroup(edp, bd, productProvenance);
  }

  void
  RunPrincipal::mergeRun(boost::shared_ptr<RunPrincipal> rp) {

    aux_.mergeAuxiliary(rp->aux());

    for (Base::const_iterator i = rp->begin(), iEnd = rp->end(); i != iEnd; ++i) {

      std::auto_ptr<Group> g(new Group());
      g->swap(**i);

      addOrReplaceGroup(g);
    }
  }

  void
  RunPrincipal::swap(RunPrincipal& iOther) {
    swapBase(iOther);
    std::swap(aux_,iOther.aux_);
  }
}
