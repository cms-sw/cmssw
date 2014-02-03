#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Utilities/interface/Algorithms.h"

namespace edm {

  std::string const LuminosityBlock::emptyString_;

  LuminosityBlock::LuminosityBlock(LuminosityBlockPrincipal& lbp, ModuleDescription const& md,
                                   ModuleCallingContext const* moduleCallingContext) :
        provRecorder_(lbp, md),
        aux_(lbp.aux()),
        run_(new Run(lbp.runPrincipal(), md, moduleCallingContext)),
        moduleCallingContext_(moduleCallingContext) {
  }

  LuminosityBlock::~LuminosityBlock() {
    // anything left here must be the result of a failure
    // let's record them as failed attempts in the event principal
    for_all(putProducts_, principal_get_adapter_detail::deleter());
  }

  LuminosityBlockIndex
  LuminosityBlock::index() const {
    return luminosityBlockPrincipal().index();
  }

  
  LuminosityBlockPrincipal&
  LuminosityBlock::luminosityBlockPrincipal() {
    return dynamic_cast<LuminosityBlockPrincipal&>(provRecorder_.principal());
  }

  void
  LuminosityBlock::setConsumer(EDConsumerBase const* iConsumer) {
    provRecorder_.setConsumer(iConsumer);
    if(run_) {
      const_cast<Run*>(run_.get())->setConsumer(iConsumer);
    }
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
  LuminosityBlock::getAllProvenance(std::vector<Provenance const*>& provenances) const {
    luminosityBlockPrincipal().getAllProvenance(provenances);
  }

  void
  LuminosityBlock::commit_() {
    LuminosityBlockPrincipal& lbp = luminosityBlockPrincipal();
    ProductPtrVec::iterator pit(putProducts().begin());
    ProductPtrVec::iterator pie(putProducts().end());

    while(pit != pie) {
        lbp.put(*pit->second, pit->first);
        // Ownership has passed, so clear the pointer.
        pit->first.reset();
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

  void
  LuminosityBlock::addToGotBranchIDs(Provenance const& prov) const {
    gotBranchIDs_.insert(prov.branchID());
  }

  BasicHandle
  LuminosityBlock::getByLabelImpl(std::type_info const&, std::type_info const& iProductType, const InputTag& iTag) const {
    BasicHandle h = provRecorder_.getByLabel_(TypeID(iProductType), iTag, moduleCallingContext_);
    if (h.isValid()) {
      addToGotBranchIDs(*(h.provenance()));
    }
    return h;
  }
}
