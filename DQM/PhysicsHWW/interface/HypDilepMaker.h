#ifndef HYPDILEPMAKER_H
#define HYPDILEPMAKER_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class HypDilepMaker {

  public:

    HypDilepMaker(){};
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

};

#endif
