#include "FWCore/Framework/interface/RunPrincipal.h"

#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/ProductHolder.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  RunPrincipal::RunPrincipal(
    boost::shared_ptr<RunAuxiliary> aux,
    boost::shared_ptr<ProductRegistry const> reg,
    ProcessConfiguration const& pc,
    HistoryAppender* historyAppender,
    unsigned int iRunIndex) :
    Base(reg, reg->productLookup(InRun), pc, InRun, historyAppender),
      aux_(aux), index_(iRunIndex), complete_(false) {
  }

  void
  RunPrincipal::fillRunPrincipal(ProcessHistoryRegistry& processHistoryRegistry, DelayedReader* reader) {
    complete_ = false;

    fillPrincipal(aux_->processHistoryID(), processHistoryRegistry, reader);

    for(auto const& prod : *this) {
      prod->setProcessHistory(processHistory());
    }
  }

  void
  RunPrincipal::put(
        BranchDescription const& bd,
        WrapperOwningHolder const& edp) {

    // Assert commented out for LHESource.
    // assert(bd.produced());
    if(!edp.isValid()) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Pointer")
        << "put: Cannot put because auto_ptr to product is null."
        << "\n";
    }
    ProductHolderBase* phb = getExistingProduct(bd.branchID());
    assert(phb);
    // ProductHolder assumes ownership
    putOrMerge(edp, phb);
  }

  void
  RunPrincipal::readImmediate() const {
    for(auto const& prod : *this) {
      ProductHolderBase const& phb = *prod;
      if(phb.singleProduct() && !phb.branchDescription().produced()) {
        if(!phb.productUnavailable()) {
          resolveProductImmediate(phb);
        }
      }
    }
  }

  void
  RunPrincipal::resolveProductImmediate(ProductHolderBase const& phb) const {
    if(phb.branchDescription().produced()) return; // nothing to do.
    if(!reader()) return; // nothing to do.

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(phb.branchDescription());
    WrapperOwningHolder edp(reader()->getProduct(bk, phb.productData().getInterface(), this));

    // Now fix up the ProductHolder
    if(edp.isValid()) {
      putOrMerge(edp, &phb);
    }
  }
}
