#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Group.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {

  LuminosityBlockPrincipal::LuminosityBlockPrincipal(
      boost::shared_ptr<LuminosityBlockAuxiliary> aux,
      boost::shared_ptr<ProductRegistry const> reg,
      ProcessConfiguration const& pc,
      boost::shared_ptr<RunPrincipal> rp) :
        Base(reg, pc, InLumi),
        runPrincipal_(rp),
        aux_(aux) {
  }

  void
  LuminosityBlockPrincipal::fillLuminosityBlockPrincipal(
      boost::shared_ptr<BranchMapper> mapper,
      boost::shared_ptr<DelayedReader> rtrv) {
    fillPrincipal(aux_->processHistoryID_, mapper, rtrv);
    if (productRegistry().productProduced(InLumi)) {
      addToProcessHistory();
    }
    mapper->processHistoryID() = processHistoryID();
  }

  void 
  LuminosityBlockPrincipal::put(
	ConstBranchDescription const& bd,
	std::auto_ptr<EDProduct> edp,
	std::auto_ptr<ProductProvenance> productProvenance) {

    assert(bd.produced());
    if (edp.get() == 0) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Pointer")
	<< "put: Cannot put because auto_ptr to product is null."
	<< "\n";
    }
    branchMapperPtr()->insert(*productProvenance);
    Group *g = getExistingGroup(bd.branchID());
    assert(g);
    // Group assumes ownership
    putOrMerge(edp, productProvenance, g);
  }

  void
  LuminosityBlockPrincipal::readImmediate() const {
    for (Principal::const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
      Group const& g = **i;
      if (!g.branchDescription().produced()) {
        if (g.provenanceAvailable()) {
	  g.resolveProvenance(branchMapperPtr());
        }
        if (!g.productUnavailable()) {
          resolveProductImmediate(g);
        }
      }
    }
    branchMapperPtr()->setDelayedRead(false);
  }

  void
  LuminosityBlockPrincipal::resolveProductImmediate(Group const& g) const {
    if (g.branchDescription().produced()) return; // nothing to do.

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(g.branchDescription());
    std::auto_ptr<EDProduct> edp(store()->getProduct(bk, this));

    // Now fix up the Group
    if (edp.get() != 0) {
      putOrMerge(edp, &g);
    }
  }

  void
  LuminosityBlockPrincipal::swap(LuminosityBlockPrincipal& iOther) {
    swapBase(iOther);
    std::swap(runPrincipal_,iOther.runPrincipal_);
    std::swap(aux_, iOther.aux_);
  }
  
}

