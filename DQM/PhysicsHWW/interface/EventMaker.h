#ifndef EVENTMAKER_H
#define EVENTMAKER_H

#include "FWCore/Framework/interface/Event.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class EventMaker {

  public:

    EventMaker();
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

};

#endif
