#include "DQM/PhysicsHWW/interface/EventMaker.h"

EventMaker::EventMaker() {}

void EventMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
     hww.Load_evt_run();
     hww.Load_evt_event();
     hww.Load_evt_lumiBlock();
     hww.Load_evt_isRealData();

     hww.evt_run()        = iEvent.id().run()       ;
     hww.evt_event()      = iEvent.id().event()     ;
     hww.evt_lumiBlock()  = iEvent.luminosityBlock();
     hww.evt_isRealData() = iEvent.isRealData()     ;

}
