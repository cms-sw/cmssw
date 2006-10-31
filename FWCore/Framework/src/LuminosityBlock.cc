#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"

namespace edm {

    LuminosityBlock::LuminosityBlock(LuminosityBlockPrincipal& dbk, ModuleDescription const& md) :
	DataViewImpl(dbk.impl(), md),
	aux_(dbk.aux()) {
    }
}
