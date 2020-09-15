#include "FWCore/Framework/interface/ProcessBlockForOutput.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"

namespace edm {
  ProcessBlockForOutput::ProcessBlockForOutput(ProcessBlockPrincipal const& pbp,
                                               ModuleDescription const& md,
                                               ModuleCallingContext const* mcc,
                                               bool isAtEnd)
      : OccurrenceForOutput(pbp, md, mcc, isAtEnd), processName_(&pbp.processName()) {}

  ProcessBlockForOutput::~ProcessBlockForOutput() {}

}  // namespace edm
