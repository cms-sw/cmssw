#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

namespace edm {

  std::string const LuminosityBlock::emptyString_;

  LuminosityBlock::LuminosityBlock(LuminosityBlockPrincipal const& lbp, ModuleDescription const& md,
                                   ModuleCallingContext const* moduleCallingContext) :
        provRecorder_(lbp, md),
        aux_(lbp.aux()),
        run_(new Run(lbp.runPrincipal(), md, moduleCallingContext)),
        moduleCallingContext_(moduleCallingContext) {
  }

  LuminosityBlock::~LuminosityBlock() {
  }

  LuminosityBlockIndex
  LuminosityBlock::index() const {
    return luminosityBlockPrincipal().index();
  }

  LuminosityBlock::CacheIdentifier_t
  LuminosityBlock::cacheIdentifier() const {return luminosityBlockPrincipal().cacheIdentifier();}

  
  void
  LuminosityBlock::setConsumer(EDConsumerBase const* iConsumer) {
    provRecorder_.setConsumer(iConsumer);
    if(run_) {
      const_cast<Run*>(run_.get())->setConsumer(iConsumer);
    }
  }
  
  void
  LuminosityBlock::setSharedResourcesAcquirer( SharedResourcesAcquirer* iResourceAcquirer) {
    provRecorder_.setSharedResourcesAcquirer(iResourceAcquirer);
    const_cast<Run*>(run_.get())->setSharedResourcesAcquirer(iResourceAcquirer);
  }


  LuminosityBlockPrincipal const&
  LuminosityBlock::luminosityBlockPrincipal() const {
    return dynamic_cast<LuminosityBlockPrincipal const&>(provRecorder_.principal());
  }

  Provenance
  LuminosityBlock::getProvenance(BranchID const& bid) const {
    return luminosityBlockPrincipal().getProvenance(bid, moduleCallingContext_);
  }

  void
  LuminosityBlock::getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const {
    luminosityBlockPrincipal().getAllStableProvenance(provenances);
  }

  void
  LuminosityBlock::commit_() {
    LuminosityBlockPrincipal const& lbp = luminosityBlockPrincipal();
    ProductPtrVec::iterator pit(putProducts().begin());
    ProductPtrVec::iterator pie(putProducts().end());

    while(pit != pie) {
        lbp.put(*pit->second, std::move(get_underlying_safe(pit->first)));
        ++pit;
    }

    // the cleanup is all or none
    putProducts().clear();
  }

  ProcessHistoryID const&
  LuminosityBlock::processHistoryID() const {
    return luminosityBlockPrincipal().processHistoryID();
  }

  ProcessHistory const&
  LuminosityBlock::processHistory() const {
    return provRecorder_.processHistory();
  }

  BasicHandle
  LuminosityBlock::getByLabelImpl(std::type_info const&, std::type_info const& iProductType, const InputTag& iTag) const {
    BasicHandle h = provRecorder_.getByLabel_(TypeID(iProductType), iTag, moduleCallingContext_);
    return h;
  }
}
