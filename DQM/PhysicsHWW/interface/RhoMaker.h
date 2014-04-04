#ifndef RHOMAKER_H
#define RHOMAKER_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class RhoMaker {

  public:

    RhoMaker(const edm::ParameterSet&, edm::ConsumesCollector);
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

  private:

    edm::EDGetTokenT<double>       Rho_;
    edm::EDGetTokenT<double>       wwRho_;
    edm::EDGetTokenT<double>       wwRhoVor_;
    edm::EDGetTokenT<double>       RhoForEGIso_;

};

#endif
