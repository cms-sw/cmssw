#ifndef PFELTOELASSMAKER_H
#define PFELTOELASSMAKER_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class PFElToElAssMaker {

  public:

    PFElToElAssMaker() {};
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

};

#endif
