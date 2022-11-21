#include "FWCore/Framework/interface/RunPrincipal.h"

#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/ProductResolverBase.h"
#include "FWCore/Framework/interface/MergeableRunProductMetadata.h"
#include "FWCore/Framework/interface/ProductPutterBase.h"
#include "FWCore/Framework/src/ProductPutOrMergerBase.h"

namespace edm {
  RunPrincipal::RunPrincipal(std::shared_ptr<ProductRegistry const> reg,
                             ProcessConfiguration const& pc,
                             HistoryAppender* historyAppender,
                             unsigned int iRunIndex,
                             bool isForPrimaryProcess,
                             MergeableRunProductProcesses const* mergeableRunProductProcesses)
      : Base(reg, reg->productLookup(InRun), pc, InRun, historyAppender, isForPrimaryProcess), index_(iRunIndex) {
    if (mergeableRunProductProcesses) {  // primary RunPrincipals of EventProcessor
      mergeableRunProductMetadataPtr_ = (std::make_unique<MergeableRunProductMetadata>(*mergeableRunProductProcesses));
    }
  }

  RunPrincipal::~RunPrincipal() {}

  void RunPrincipal::fillRunPrincipal(ProcessHistoryRegistry const& processHistoryRegistry, DelayedReader* reader) {
    m_reducedHistoryID = processHistoryRegistry.reducedProcessHistoryID(aux_.processHistoryID());
    auto history = processHistoryRegistry.getMapped(aux_.processHistoryID());
    fillPrincipal(aux_.processHistoryID(), history, reader);

    for (auto& prod : *this) {
      prod->setMergeableRunProductMetadata(mergeableRunProductMetadataPtr_.get());
    }
  }

  void RunPrincipal::put(BranchDescription const& bd, std::unique_ptr<WrapperBase> edp) const {
    put_(bd, std::move(edp));
  }

  void RunPrincipal::put(ProductResolverIndex index, std::unique_ptr<WrapperBase> edp) const {
    auto phb = getProductResolverByIndex(index);
    dynamic_cast<ProductPutterBase const*>(phb)->putProduct(std::move(edp));
  }

  void RunPrincipal::putOrMerge(BranchDescription const& bd, std::unique_ptr<WrapperBase> prod) const {
    if (prod.get() == nullptr) {
      throw edm::Exception(edm::errors::InsertFailure, "Null Pointer")
          << "putOrMerge: Cannot put because unique_ptr to product is null."
          << "\n";
    }
    auto phb = getExistingProduct(bd.branchID());
    assert(phb);
    dynamic_cast<ProductPutOrMergerBase const*>(phb)->putOrMergeProduct(std::move(prod));
  }

  unsigned int RunPrincipal::transitionIndex_() const { return index().value(); }

  void RunPrincipal::preReadFile() {
    if (mergeableRunProductMetadataPtr_) {
      mergeableRunProductMetadataPtr_->preReadFile();
    }
  }

}  // namespace edm
