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
    if (group != 0) {

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
    } else {
      addGroup_(g);
    }
  }

  void
  LuminosityBlockPrincipal::addGroup(ConstBranchDescription const& bd) {
    std::auto_ptr<Group> g(new Group(bd, ProductID()));
    addOrReplaceGroup(g);
  }

  void
  LuminosityBlockPrincipal::addGroup(boost::shared_ptr<EDProduct> prod,
	ConstBranchDescription const& bd,
	std::auto_ptr<ProductProvenance> productProvenance) {
    std::auto_ptr<Group> g(new Group(prod, bd, ProductID(), productProvenance));
    addOrReplaceGroup(g);
  }

  void
  LuminosityBlockPrincipal::addGroup(ConstBranchDescription const& bd,
	std::auto_ptr<ProductProvenance> productProvenance) {
    std::auto_ptr<Group> g(new Group(bd, ProductID(), productProvenance));
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
    this->addGroup(edp, bd, productProvenance);
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

