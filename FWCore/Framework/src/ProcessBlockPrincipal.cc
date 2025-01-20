#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/ProductPutterBase.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchType.h"

#include <utility>

namespace edm {

  ProcessBlockPrincipal::ProcessBlockPrincipal(std::shared_ptr<ProductRegistry const> reg,
                                               std::vector<std::shared_ptr<ProductResolverBase>>&& resolvers,
                                               ProcessConfiguration const& pc)
      : Principal(reg, std::move(resolvers), pc, InProcess, nullptr) {}

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
