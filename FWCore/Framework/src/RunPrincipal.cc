#include "FWCore/Framework/interface/RunPrincipal.h"

#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/ProductHolder.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  RunPrincipal::RunPrincipal(
    std::shared_ptr<RunAuxiliary> aux,
    std::shared_ptr<ProductRegistry const> reg,
    ProcessConfiguration const& pc,
    HistoryAppender* historyAppender,
    unsigned int iRunIndex) :
    Base(reg, reg->productLookup(InRun), pc, InRun, historyAppender),
      aux_(aux), index_(iRunIndex), complete_(false) {
  }

  void
  RunPrincipal::fillRunPrincipal(ProcessHistoryRegistry const& processHistoryRegistry, DelayedReader* reader) {
    complete_ = false;

    m_reducedHistoryID = processHistoryRegistry.reducedProcessHistoryID(aux_->processHistoryID());
    fillPrincipal(aux_->processHistoryID(), processHistoryRegistry, reader);

    for(auto const& prod : *this) {
      prod->setProcessHistory(processHistory());
    }
  }

  void
  RunPrincipal::put(
        BranchDescription const& bd,
        std::unique_ptr<WrapperBase>  edp) {

    // Assert commented out for LHESource.
    // assert(bd.produced());
    if(edp.get() == nullptr) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Pointer")
        << "put: Cannot put because unique_ptr to product is null."
        << "\n";
    }
    ProductHolderBase* phb = getExistingProduct(bd.branchID());
    assert(phb);
    // ProductHolder assumes ownership
    putOrMerge(std::move(edp), phb);
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
    std::unique_ptr<WrapperBase> edp(reader()->getProduct(bk, this));

    // Now fix up the ProductHolder
    if(edp.get() != nullptr) {
      putOrMerge(std::move(edp), &phb);
    }
  }

  unsigned int
  RunPrincipal::transitionIndex_() const {
    return index().value();
  }

}
