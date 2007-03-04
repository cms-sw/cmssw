#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

    Event::Event(EventPrincipal& dbk, ModuleDescription const& md) :
	DataViewImpl(dbk.groupGetter(), md, InEvent),
	aux_(dbk.aux()) {
    }

    LuminosityBlockNumber_t
    Event::luminosityBlock() const {
      return getLuminosityBlock().luminosityBlock();
    }

    Run const&
    Event::getRun() const {
      return getLuminosityBlock().getRun();
    }

    RunNumber_t
    Event::run() const {
      return getRun().run();
    }
}
