#include "FWCore/Framework/interface/RunForOutput.h"

#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

namespace edm {

  RunForOutput::RunForOutput(RunTransitionInfo const& info,
                             ModuleDescription const& md,
                             ModuleCallingContext const* mcc,
                             bool isAtEnd,
                             MergeableRunProductMetadata const* mrpm)
      : RunForOutput(info.principal(), md, mcc, isAtEnd, mrpm) {}

  RunForOutput::RunForOutput(RunPrincipal const& rp,
                             ModuleDescription const& md,
                             ModuleCallingContext const* moduleCallingContext,
                             bool isAtEnd,
                             MergeableRunProductMetadata const* mrpm)
      : OccurrenceForOutput(rp, md, moduleCallingContext, isAtEnd),
        aux_(rp.aux()),
        mergeableRunProductMetadata_(mrpm) {}

  RunForOutput::~RunForOutput() {}

  RunPrincipal const& RunForOutput::runPrincipal() const { return dynamic_cast<RunPrincipal const&>(principal()); }

  /**\return Reusable index which can be used to separate data for different simultaneous Runs.
   */
  RunIndex RunForOutput::index() const { return runPrincipal().index(); }

}  // namespace edm
