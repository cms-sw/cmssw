#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DPGAnalysis/SiStripTools/interface/TinyEvent.h"


TinyEvent::TinyEvent(const edm::Event& event):
  _event(event.id().event()),
  _orbit(event.orbitNumber() < 0 ? 0 : event.orbitNumber()),
  _bx(event.bunchCrossing() < 0 ? 0 : event.bunchCrossing()) { }

TinyEvent::TinyEvent(const edm::EventAuxiliary& eaux):
  _event(eaux.event()),
  _orbit(eaux.orbitNumber() < 0 ? 0 : eaux.orbitNumber()),
  _bx(eaux.bunchCrossing() < 0 ? 0 : eaux.bunchCrossing()) { }


