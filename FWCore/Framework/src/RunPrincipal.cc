#include "FWCore/Framework/interface/RunPrincipal.h"

#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/ProductResolverBase.h"

namespace edm {
  RunPrincipal::RunPrincipal(
    std::shared_ptr<RunAuxiliary> aux,
    std::shared_ptr<ProductRegistry const> reg,
    ProcessConfiguration const& pc,
    HistoryAppender* historyAppender,
    unsigned int iRunIndex,
    bool isForPrimaryProcess) :
    Base(reg, reg->productLookup(InRun), pc, InRun, historyAppender, isForPrimaryProcess),
      aux_(aux), index_(iRunIndex), complete_(false) {
  }

  void
  RunPrincipal::fillRunPrincipal(ProcessHistoryRegistry const& processHistoryRegistry, DelayedReader* reader) {
    complete_ = false;

    m_reducedHistoryID = processHistoryRegistry.reducedProcessHistoryID(aux_->processHistoryID());
    fillPrincipal(aux_->processHistoryID(), processHistoryRegistry, reader);

    for(auto& prod : *this) {
      prod->setProcessHistory(processHistory());
    }
  }

  void
  RunPrincipal::put(
        BranchDescription const& bd,
        std::unique_ptr<WrapperBase>  edp) const {
    putOrMerge(bd,std::move(edp));
  }

  void
  RunPrincipal::put(ProductResolverIndex index,
                    std::unique_ptr<WrapperBase> edp) const {
    auto phb = getProductResolverByIndex(index);
    phb->putOrMergeProduct(std::move(edp));
  }
  
  unsigned int
  RunPrincipal::transitionIndex_() const {
    return index().value();
  }

}
