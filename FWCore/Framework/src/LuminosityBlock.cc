#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/Run.h"

namespace edm {

    LuminosityBlock::LuminosityBlock(LuminosityBlockPrincipal& dbk, ModuleDescription const& md) :
	DataViewImpl(dbk.groupGetter(), md, InLumi),
	aux_(dbk.aux()) {
    }

    RunNumber_t
    LuminosityBlock::run() const {
      return getRun().run();
    }
}
