#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/ProductPutterBase.h"

namespace edm {

  ProcessBlock::ProcessBlock(ProcessBlockPrincipal const& pbp,
                             ModuleDescription const& md,
                             ModuleCallingContext const* moduleCallingContext,
                             bool isAtEnd)
      : provRecorder_(pbp, md, isAtEnd), moduleCallingContext_(moduleCallingContext) {}

  void ProcessBlock::setProducer(ProducerBase const* iProducer) {
    provRecorder_.setProducer(iProducer);
    //set appropriate size
    putProducts_.resize(provRecorder_.putTokenIndexToProductResolverIndex().size());
  }

  ProcessBlock::CacheIdentifier_t ProcessBlock::cacheIdentifier() const {
    return processBlockPrincipal().cacheIdentifier();
  }

  std::string const& ProcessBlock::processName() const { return processBlockPrincipal().processName(); }

  ProcessBlockPrincipal const& ProcessBlock::processBlockPrincipal() const {
    return dynamic_cast<ProcessBlockPrincipal const&>(provRecorder_.principal());
  }

  void ProcessBlock::commit_(std::vector<edm::ProductResolverIndex> const& iShouldPut) {
    ProcessBlockPrincipal const& pbp = processBlockPrincipal();
    size_t nPut = 0;
    for (size_t i = 0; i < putProducts().size(); ++i) {
      auto& product = get_underlying_safe(putProducts()[i]);
      if (product) {
        pbp.put(provRecorder_.putTokenIndexToProductResolverIndex()[i], std::move(product));
        ++nPut;
      }
    }

    auto sz = iShouldPut.size();
    if (sz != 0 and sz != nPut) {
      //some were missed
      auto& principal = provRecorder_.principal();
      for (auto index : iShouldPut) {
        auto resolver = principal.getProductResolverByIndex(index);
        if (not resolver->productResolved() and isEndTransition(provRecorder_.transition()) ==
                                                    resolver->branchDescription().availableOnlyAtEndTransition()) {
          dynamic_cast<ProductPutterBase const*>(resolver)->putProduct(std::unique_ptr<WrapperBase>());
        }
      }
    }

    // the cleanup is all or none
    putProducts().clear();
  }

}  // namespace edm
