#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/Run.h"

namespace edm {

    LuminosityBlock::LuminosityBlock(LuminosityBlockPrincipal& dbk, ModuleDescription const& md) :
	DataViewImpl(dbk.groupGetter(), md, InLumi),
	aux_(dbk.aux()),
	run_(new Run(dbk.runPrincipal(), md)) {
    }
}
