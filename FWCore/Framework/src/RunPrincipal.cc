#include "FWCore/Framework/interface/RunPrincipal.h"

#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/Group.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  RunPrincipal::RunPrincipal(
    boost::shared_ptr<RunAuxiliary> aux,
    boost::shared_ptr<ProductRegistry const> reg,
    ProcessConfiguration const& pc,
    HistoryAppender* historyAppender) :
    Base(reg, pc, InRun, historyAppender),
      aux_(aux) {
  }

  void
  RunPrincipal::fillRunPrincipal(
    boost::shared_ptr<BranchMapper> mapper,
    DelayedReader* reader) {

    fillPrincipal(aux_->processHistoryID(), mapper, reader);

    branchMapperPtr()->processHistoryID() = processHistoryID();
    for (const_iterator i = this->begin(), iEnd = this->end(); i != iEnd; ++i) {
      (*i)->setProvenance(branchMapperPtr());
    }
  }

  void
  RunPrincipal::put(
        ConstBranchDescription const& bd,
        WrapperOwningHolder const& edp,
        ProductProvenance& productProvenance) {

    assert(bd.produced());
    if(!edp.isValid()) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Pointer")
        << "put: Cannot put because auto_ptr to product is null."
        << "\n";
    }
    branchMapperPtr()->insert(productProvenance);
    Group *g = getExistingGroup(bd.branchID());
    assert(g);
    // Group assumes ownership
    putOrMerge(edp, productProvenance, g);
  }

  void
  RunPrincipal::readImmediate() const {
    for (Principal::const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
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
  RunPrincipal::resolveProductImmediate(Group const& g) const {
    if(g.branchDescription().produced()) return; // nothing to do.
    if(!reader()) return; // nothing to do.

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(g.branchDescription());
    WrapperOwningHolder edp(reader()->getProduct(bk, g.productData().getInterface(), this));

    // Now fix up the Group
    if(edp.isValid()) {
      putOrMerge(edp, &g);
    }
  }
}
