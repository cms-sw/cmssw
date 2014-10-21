#ifndef PFMETMAKER_H
#define PFMETMAKER_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class PFMETMaker {

  public:

    PFMETMaker(const edm::ParameterSet&, edm::ConsumesCollector);
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

  private:

    edm::EDGetTokenT<edm::View<reco::PFMET> >       PFMET_;

};

#endif
