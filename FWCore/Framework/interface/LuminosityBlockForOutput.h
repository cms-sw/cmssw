#ifndef FWCore_Framework_LuminosityBlockForOutput_h
#define FWCore_Framework_LuminosityBlockForOutput_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     LuminosityBlockForOutput
//
/**\class LuminosityBlockForOutput LuminosityBlockForOutput.h FWCore/Framework/interface/LuminosityBlockForOutput.h

Description: This is the primary interface for accessing per luminosity block EDProducts
and inserting new derived per luminosity block EDProducts.

For its usage, see "FWCore/Framework/interface/PrincipalGetAdapter.h"

*/
/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OccurrenceForOutput.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

namespace edmtest {
  class TestOutputModule;
}

namespace edm {
  class ModuleCallingContext;

  class LuminosityBlockForOutput : public OccurrenceForOutput {
  public:
    LuminosityBlockForOutput(LumiTransitionInfo const&,
                             ModuleDescription const&,
                             ModuleCallingContext const*,
                             bool isAtEnd);
    LuminosityBlockForOutput(LuminosityBlockPrincipal const&,
                             ModuleDescription const&,
                             ModuleCallingContext const*,
                             bool isAtEnd);
    ~LuminosityBlockForOutput() override;

    LuminosityBlockAuxiliary const& luminosityBlockAuxiliary() const { return aux_; }
    LuminosityBlockID const& id() const { return aux_.id(); }
    LuminosityBlockNumber_t luminosityBlock() const { return aux_.luminosityBlock(); }
    RunNumber_t run() const { return aux_.run(); }
    Timestamp const& beginTime() const { return aux_.beginTime(); }
    Timestamp const& endTime() const { return aux_.endTime(); }

    /**\return Reusable index which can be used to separate data for different simultaneous LuminosityBlocks.
     */
    LuminosityBlockIndex index() const;

    RunForOutput const& getRun() const { return *run_; }

  private:
    friend class edmtest::TestOutputModule;  // For testing

    LuminosityBlockPrincipal const& luminosityBlockPrincipal() const;

    LuminosityBlockAuxiliary const& aux_;
    std::shared_ptr<RunForOutput const> const run_;
  };

}  // namespace edm
#endif
