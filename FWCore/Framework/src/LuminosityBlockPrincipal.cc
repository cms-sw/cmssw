#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Group.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {

  LuminosityBlockPrincipal::LuminosityBlockPrincipal(LuminosityBlockAuxiliary const& aux,
	boost::shared_ptr<ProductRegistry const> reg,
	ProcessConfiguration const& pc,
	boost::shared_ptr<BranchMapper> mapper, 
	boost::shared_ptr<DelayedReader> rtrv) :
	  Base(reg, pc, InLumi, aux.processHistoryID_, mapper, rtrv),
	  runPrincipal_(),
          aux_(aux) {
      if (reg->productProduced(InLumi)) {
        addToProcessHistory();
      }
      mapper->processHistoryID() = processHistoryID();
  }

  void
  LuminosityBlockPrincipal::addOrReplaceGroup(std::auto_ptr<Group> g) {

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
         //PrincipalCache holds onto the 'newest' version of a LumiPrincipal for a given lumi
         // but our behavior is to keep the 'old' group and merge in the new one because if there
         // is no way to merge we keep the 'old' group
         edm::swap(*group,*g);
      }
      group->mergeGroup(g.get());
    }
  }

  void
  LuminosityBlockPrincipal::addGroupScheduled(ConstBranchDescription const& bd) {
    std::auto_ptr<Group> g(new Group(bd, ProductID(), productstatus::producerNotRun()));
    addGroupOrNoThrow(g);
  }

  void
  LuminosityBlockPrincipal::addGroupSource(ConstBranchDescription const& bd) {
    std::auto_ptr<Group> g(new Group(bd, ProductID(), productstatus::producerDidNotPutProduct()));
    addGroupOrNoThrow(g);
  }

  void
  LuminosityBlockPrincipal::addGroupIfNeeded(ConstBranchDescription const& bd) {
    if (getExistingGroup(bd.branchID()) == 0) {
      addGroup(bd, true);
    }
  }

  void
  LuminosityBlockPrincipal::addGroup(ConstBranchDescription const& bd, bool dropped) {
    std::auto_ptr<Group> g(new Group(bd, ProductID(), dropped));
    addOrReplaceGroup(g);
  }

  void
  LuminosityBlockPrincipal::addToGroup(boost::shared_ptr<EDProduct> prod,
	ConstBranchDescription const& bd,
	std::auto_ptr<ProductProvenance> productProvenance) {
    std::auto_ptr<Group> g(new Group(prod, bd, ProductID(), productProvenance));
    addOrReplaceGroup(g);
  }

  void 
  LuminosityBlockPrincipal::put(boost::shared_ptr<EDProduct> edp,
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
  LuminosityBlockPrincipal::mergeLuminosityBlock(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {

    aux_.mergeAuxiliary(lbp->aux());

    for (Base::const_iterator i = lbp->begin(), iEnd = lbp->end(); i != iEnd; ++i) {
 
      std::auto_ptr<Group> g(new Group());
      g->swap(*(*i));

      addOrReplaceGroup(g);
    }
  }

  void
  LuminosityBlockPrincipal::swap(LuminosityBlockPrincipal& iOther) {
    swapBase(iOther);
    std::swap(runPrincipal_,iOther.runPrincipal_);
    std::swap(aux_,iOther.aux_);
  }
}

