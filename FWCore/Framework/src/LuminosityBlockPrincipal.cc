#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/Group.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/EDMException.h"

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

    fillPrincipal(aux_->processHistoryID(), mapper, rtrv);
    if(runPrincipal_) {
      setProcessHistory(*runPrincipal_);
    }
    branchMapperPtr()->processHistoryID() = processHistoryID();
    for(const_iterator i = this->begin(), iEnd = this->end(); i != iEnd; ++i) {
      (*i)->setProvenance(branchMapperPtr());
    }
  }

  void
  LuminosityBlockPrincipal::put(
        ConstBranchDescription const& bd,
        WrapperHolder const& edp,
        std::auto_ptr<ProductProvenance> productProvenance) {

    assert(bd.produced());
    if(!edp.isValid()) {
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
    for(Principal::const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
      Group const& g = **i;
      if(!g.branchDescription().produced()) {
        if(!g.productUnavailable()) {
          resolveProductImmediate(g);
        }
      }
    }
    branchMapperPtr()->setDelayedRead(false);
  }

  void
  LuminosityBlockPrincipal::resolveProductImmediate(Group const& g) const {
    if(g.branchDescription().produced()) return; // nothing to do.

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(g.branchDescription());
    WrapperHolder edp(store()->getProduct(bk, g.productData().getInterface(), this));

    // Now fix up the Group
    if(edp.isValid()) {
      putOrMerge(edp, &g);
    }
  }
}
