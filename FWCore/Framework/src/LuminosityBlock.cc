#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

namespace edm {

  std::string const LuminosityBlock::emptyString_;

  LuminosityBlock::LuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                   ModuleDescription const& md,
                                   ModuleCallingContext const* moduleCallingContext,
                                   bool isAtEnd)
      : provRecorder_(lbp, md, isAtEnd), aux_(lbp.aux()), run_(), moduleCallingContext_(moduleCallingContext) {}

  LuminosityBlock::~LuminosityBlock() {}

  LuminosityBlockIndex LuminosityBlock::index() const { return luminosityBlockPrincipal().index(); }

  LuminosityBlock::CacheIdentifier_t LuminosityBlock::cacheIdentifier() const {
    return luminosityBlockPrincipal().cacheIdentifier();
  }

  void LuminosityBlock::setConsumer(EDConsumerBase const* iConsumer) {
    provRecorder_.setConsumer(iConsumer);
    if (run_) {
      run_->setConsumer(iConsumer);
    }
  }

  void LuminosityBlock::setSharedResourcesAcquirer(SharedResourcesAcquirer* iResourceAcquirer) {
    provRecorder_.setSharedResourcesAcquirer(iResourceAcquirer);
    if (run_) {
      run_->setSharedResourcesAcquirer(iResourceAcquirer);
    }
  }

  void LuminosityBlock::fillRun() const {
    run_.emplace(
        luminosityBlockPrincipal().runPrincipal(), provRecorder_.moduleDescription(), moduleCallingContext_, false);
    run_->setSharedResourcesAcquirer(provRecorder_.getSharedResourcesAcquirer());
    run_->setConsumer(provRecorder_.getConsumer());
  }

  void LuminosityBlock::setProducer(ProducerBase const* iProducer) {
    provRecorder_.setProducer(iProducer);
    //set appropriate size
    putProducts_.resize(provRecorder_.putTokenIndexToProductResolverIndex().size());
  }

  LuminosityBlockPrincipal const& LuminosityBlock::luminosityBlockPrincipal() const {
    return dynamic_cast<LuminosityBlockPrincipal const&>(provRecorder_.principal());
  }

  Provenance LuminosityBlock::getProvenance(BranchID const& bid) const {
    return luminosityBlockPrincipal().getProvenance(bid, moduleCallingContext_);
  }

  void LuminosityBlock::getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const {
    luminosityBlockPrincipal().getAllStableProvenance(provenances);
  }

  void LuminosityBlock::commit_(std::vector<edm::ProductResolverIndex> const& iShouldPut) {
    LuminosityBlockPrincipal const& lbp = luminosityBlockPrincipal();
    size_t nPut = 0;
    for (size_t i = 0; i < putProducts().size(); ++i) {
      auto& p = get_underlying_safe(putProducts()[i]);
      if (p) {
        lbp.put(provRecorder_.putTokenIndexToProductResolverIndex()[i], std::move(p));
        ++nPut;
      }
    }

    auto sz = iShouldPut.size();
    if (sz != 0 and sz != nPut) {
      //some were missed
      auto& p = provRecorder_.principal();
      for (auto index : iShouldPut) {
        auto resolver = p.getProductResolverByIndex(index);
        if (not resolver->productResolved() and isEndTransition(provRecorder_.transition()) ==
                                                    resolver->branchDescription().availableOnlyAtEndTransition()) {
          resolver->putProduct(std::unique_ptr<WrapperBase>());
        }
      }
    }

    // the cleanup is all or none
    putProducts().clear();
  }

  ProcessHistoryID const& LuminosityBlock::processHistoryID() const {
    return luminosityBlockPrincipal().processHistoryID();
  }

  ProcessHistory const& LuminosityBlock::processHistory() const { return provRecorder_.processHistory(); }

  BasicHandle LuminosityBlock::getByLabelImpl(std::type_info const&,
                                              std::type_info const& iProductType,
                                              const InputTag& iTag) const {
    BasicHandle h = provRecorder_.getByLabel_(TypeID(iProductType), iTag, moduleCallingContext_);
    return h;
  }
}  // namespace edm
