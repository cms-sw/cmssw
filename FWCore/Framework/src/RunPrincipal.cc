#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Group.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

namespace edm {
  RunPrincipal::RunPrincipal(
    boost::shared_ptr<RunAuxiliary> aux,
    boost::shared_ptr<ProductRegistry const> reg,
    ProcessConfiguration const& pc) :
      Base(reg, pc, InRun),
      aux_(aux) {
  }

  void
  RunPrincipal::fillRunPrincipal(
    boost::shared_ptr<BranchMapper> mapper,
    boost::shared_ptr<DelayedReader> rtrv) {
    fillPrincipal(aux_->processHistoryID_, mapper, rtrv);
    if (productRegistry().productProduced(InRun)) {
      addToProcessHistory();
    }
    mapper->processHistoryID() = processHistoryID();
  }

  void
  RunPrincipal::fillFrom(RunPrincipal& rp) {
     fillRunPrincipal(rp.branchMapperPtr(), rp.store());
     mergeAuxiliary(rp.aux());
     for(const_iterator i = rp.begin(), iEnd = rp.end(); i != iEnd; ++i) {
       Group& group = **i;
       Group *g = getExistingGroup(group);
       assert(g);
       g->swap(group);
     }
  }

  void 
  RunPrincipal::put(
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
    checkUniquenessAndType(edp, g);
    // Group assumes ownership
    g->putOrMergeProduct(edp, productProvenance);
  }

  void
  RunPrincipal::readImmediate() const {
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
  RunPrincipal::resolveProductImmediate(Group const& g) const {
    if (g.branchDescription().produced()) return; // nothing to do.

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(g.branchDescription());
    std::auto_ptr<EDProduct> edp(store()->getProduct(bk, this));

    // Now fix up the Group
    if (edp.get() != 0) {
      checkUniquenessAndType(edp, &g);
      g.putOrMergeProduct(edp);
    }
  }

  void
  RunPrincipal::swap(RunPrincipal& iOther) {
    swapBase(iOther);
    std::swap(aux_, iOther.aux_);
  }
}
