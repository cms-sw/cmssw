#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/ProductHolder.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  LuminosityBlockPrincipal::LuminosityBlockPrincipal(
      std::shared_ptr<LuminosityBlockAuxiliary> aux,
      std::shared_ptr<ProductRegistry const> reg,
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
      ProcessHistoryRegistry const& processHistoryRegistry,
      DelayedReader* reader) {

    complete_ = false;

    fillPrincipal(aux_->processHistoryID(), processHistoryRegistry, reader);

    for(auto& prod : *this) {
      prod->setProcessHistory(processHistory());
    }
  }

  void
  LuminosityBlockPrincipal::put(
        BranchDescription const& bd,
        std::unique_ptr<WrapperBase> edp) const {
    putOrMerge(bd,std::move(edp));
  }

  unsigned int
  LuminosityBlockPrincipal::transitionIndex_() const {
    return index().value();
  }

}
