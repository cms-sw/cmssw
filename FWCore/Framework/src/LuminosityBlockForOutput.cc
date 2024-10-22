#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"

#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

namespace edm {

  LuminosityBlockForOutput::LuminosityBlockForOutput(LumiTransitionInfo const& info,
                                                     ModuleDescription const& md,
                                                     ModuleCallingContext const* mcc,
                                                     bool isAtEnd)
      : LuminosityBlockForOutput(info.principal(), md, mcc, isAtEnd) {}

  LuminosityBlockForOutput::LuminosityBlockForOutput(LuminosityBlockPrincipal const& lbp,
                                                     ModuleDescription const& md,
                                                     ModuleCallingContext const* moduleCallingContext,
                                                     bool isAtEnd)
      : OccurrenceForOutput(lbp, md, moduleCallingContext, isAtEnd),
        aux_(lbp.aux()),
        run_(new RunForOutput(lbp.runPrincipal(), md, moduleCallingContext, false)) {}

  LuminosityBlockForOutput::~LuminosityBlockForOutput() {}

  LuminosityBlockPrincipal const& LuminosityBlockForOutput::luminosityBlockPrincipal() const {
    return dynamic_cast<LuminosityBlockPrincipal const&>(principal());
  }

  /**\return Reusable index which can be used to separate data for different simultaneous LuminosityBlocks.
   */
  LuminosityBlockIndex LuminosityBlockForOutput::index() const { return luminosityBlockPrincipal().index(); }

}  // namespace edm
