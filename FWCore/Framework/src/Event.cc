#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

    Event::Event(EventPrincipal& dbk, ModuleDescription const& md) :
	DataViewImpl(dbk.impl(), md, InEvent),
	aux_(dbk.aux()) {
    }
}
