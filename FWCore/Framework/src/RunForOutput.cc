#include "FWCore/Framework/interface/RunForOutput.h"

#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

namespace edm {

  RunForOutput::RunForOutput(RunPrincipal const& rp, ModuleDescription const& md,
                             ModuleCallingContext const* moduleCallingContext)
      : OccurrenceForOutput(rp, md, moduleCallingContext), aux_(rp.aux()) {}

  RunForOutput::~RunForOutput() {}

  RunPrincipal const& RunForOutput::runPrincipal() const { return dynamic_cast<RunPrincipal const&>(principal()); }
}
