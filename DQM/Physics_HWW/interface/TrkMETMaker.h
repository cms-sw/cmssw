#ifndef TRKMETMAKER_H
#define TRKMETMAKER_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/Physics_HWW/interface/HWW.h"

class TrkMETMaker {

  public:

    TrkMETMaker() {};
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

};

#endif

