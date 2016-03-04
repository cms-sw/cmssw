#include "FWCore/Framework/interface/RunForOutput.h"

#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

namespace edm {

  RunForOutput::RunForOutput(RunPrincipal const& rp, ModuleDescription const& md,
           ModuleCallingContext const* moduleCallingContext) :
        provRecorder_(rp, md),
        aux_(rp.aux()),
        moduleCallingContext_(moduleCallingContext)  {
  }

  RunForOutput::~RunForOutput() {
  }

  RunPrincipal const&
  RunForOutput::runPrincipal() const {
    return dynamic_cast<RunPrincipal const&>(provRecorder_.principal());
  }

  Provenance
  RunForOutput::getProvenance(BranchID const& bid) const {
    return runPrincipal().getProvenance(bid, moduleCallingContext_);
  }

  void
  RunForOutput::getAllProvenance(std::vector<Provenance const*>& provenances) const {
    runPrincipal().getAllProvenance(provenances);
  }

  bool
  RunForOutput::getByToken(EDGetToken token, TypeID const& typeID, BasicHandle& result) const {
    result.clear();
    result = provRecorder_.getByToken_(typeID, PRODUCT_TYPE, token, moduleCallingContext_);
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  ProcessHistoryID const&
  RunForOutput::processHistoryID() const {
    return runPrincipal().processHistoryID();
  }

  ProcessHistory const&
  RunForOutput::processHistory() const {
    return provRecorder_.processHistory();
  }
}
