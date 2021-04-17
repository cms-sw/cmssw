#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/src/ProductPutterBase.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchType.h"

#include <utility>

namespace edm {

  ProcessBlockPrincipal::ProcessBlockPrincipal(std::shared_ptr<ProductRegistry const> reg,
                                               ProcessConfiguration const& pc,
                                               bool isForPrimaryProcess)
      : Principal(reg, reg->productLookup(InProcess), pc, InProcess, nullptr, isForPrimaryProcess) {}

  void ProcessBlockPrincipal::fillProcessBlockPrincipal(std::string const& processNameOfBlock, DelayedReader* reader) {
    processName_ = processNameOfBlock;
    fillPrincipal(processNameOfBlock, reader);
  }

  void ProcessBlockPrincipal::put(ProductResolverIndex index, std::unique_ptr<WrapperBase> edp) const {
    auto phb = getProductResolverByIndex(index);
    dynamic_cast<ProductPutterBase const*>(phb)->putProduct(std::move(edp));
  }

  unsigned int ProcessBlockPrincipal::transitionIndex_() const {
    // Concurrent ProcessBlocks does not make any sense so just always
    // return 0 here.
    return 0;
  }

}  // namespace edm
