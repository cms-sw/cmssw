#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/ProductHolder.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  LuminosityBlockPrincipal::LuminosityBlockPrincipal(
      boost::shared_ptr<LuminosityBlockAuxiliary> aux,
      boost::shared_ptr<ProductRegistry const> reg,
      ProcessConfiguration const& pc,
      HistoryAppender* historyAppender,
      unsigned int index) :
    Base(reg, reg->productLookup(InLumi), pc, InLumi, historyAppender),
        runPrincipal_(),
        aux_(aux),
        index_(index),
        complete_(false) {
  }

  void
  LuminosityBlockPrincipal::fillLuminosityBlockPrincipal(
      ProcessHistoryRegistry& processHistoryRegistry,
      DelayedReader* reader) {

    complete_ = false;

    fillPrincipal(aux_->processHistoryID(), processHistoryRegistry, reader);

    for(auto const& prod : *this) {
      prod->setProcessHistory(processHistory());
    }
  }

  void
  LuminosityBlockPrincipal::put(
        BranchDescription const& bd,
        WrapperOwningHolder const& edp) {

    assert(bd.produced());
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
  LuminosityBlockPrincipal::readImmediate() const {
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
  LuminosityBlockPrincipal::resolveProductImmediate(ProductHolderBase const& phb) const {
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

  unsigned int
  LuminosityBlockPrincipal::transitionIndex_() const {
    return index().value();
  }

}
