#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"

#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

namespace edm {

  LuminosityBlockForOutput::LuminosityBlockForOutput(LuminosityBlockPrincipal const& lbp, ModuleDescription const& md,
                                   ModuleCallingContext const* moduleCallingContext) :
        provRecorder_(lbp, md),
        aux_(lbp.aux()),
        run_(new RunForOutput(lbp.runPrincipal(), md, moduleCallingContext)),
        moduleCallingContext_(moduleCallingContext) {
  }

  LuminosityBlockForOutput::~LuminosityBlockForOutput() {
  }

  void
  LuminosityBlockForOutput::setConsumer(EDConsumerBase const* iConsumer) {
    provRecorder_.setConsumer(iConsumer);
    if(run_) {
      const_cast<RunForOutput*>(run_.get())->setConsumer(iConsumer);
    }
  }
  
  LuminosityBlockPrincipal const&
  LuminosityBlockForOutput::luminosityBlockPrincipal() const {
    return dynamic_cast<LuminosityBlockPrincipal const&>(provRecorder_.principal());
  }

  Provenance
  LuminosityBlockForOutput::getProvenance(BranchID const& bid) const {
    return luminosityBlockPrincipal().getProvenance(bid, moduleCallingContext_);
  }

  void
  LuminosityBlockForOutput::getAllProvenance(std::vector<Provenance const*>& provenances) const {
    luminosityBlockPrincipal().getAllProvenance(provenances);
  }

  bool
  LuminosityBlockForOutput::getByToken(EDGetToken token, TypeID const& typeID, BasicHandle& result) const {
    result.clear();
    result = provRecorder_.getByToken_(typeID, PRODUCT_TYPE, token, moduleCallingContext_);
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  ProcessHistoryID const&
  LuminosityBlockForOutput::processHistoryID() const {
    return luminosityBlockPrincipal().processHistoryID();
  }

  ProcessHistory const&
  LuminosityBlockForOutput::processHistory() const {
    return provRecorder_.processHistory();
  }

}
