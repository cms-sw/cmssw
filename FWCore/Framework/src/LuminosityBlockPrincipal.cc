#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ProductPutterBase.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

namespace edm {
  LuminosityBlockPrincipal::LuminosityBlockPrincipal(std::shared_ptr<ProductRegistry const> reg,
                                                     std::vector<std::shared_ptr<ProductResolverBase>>&& resolvers,
                                                     ProcessConfiguration const& pc,
                                                     HistoryAppender* historyAppender,
                                                     unsigned int index)
      : Base(reg, std::move(resolvers), pc, InLumi, historyAppender), runPrincipal_(), index_(index) {}

  void LuminosityBlockPrincipal::fillLuminosityBlockPrincipal(ProcessHistory const* processHistory,
                                                              DelayedReader* reader) {
    fillPrincipal(aux_.processHistoryID(), processHistory, reader);
  }

  void LuminosityBlockPrincipal::put(BranchDescription const& bd, std::unique_ptr<WrapperBase> edp) const {
    put_(bd, std::move(edp));
  }

  void LuminosityBlockPrincipal::put(ProductResolverIndex index, std::unique_ptr<WrapperBase> edp) const {
    auto phb = getProductResolverByIndex(index);
    dynamic_cast<ProductPutterBase const*>(phb)->putProduct(std::move(edp));
  }

  unsigned int LuminosityBlockPrincipal::transitionIndex_() const { return index().value(); }

}  // namespace edm
