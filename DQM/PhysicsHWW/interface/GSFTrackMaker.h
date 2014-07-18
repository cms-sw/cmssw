#ifndef GSFTRACKMAKER_H
#define GSFTRACKMAKER_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class GSFTrackMaker {

  public:

    GSFTrackMaker(const edm::ParameterSet&, edm::ConsumesCollector);
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

  private:

    edm::EDGetTokenT<edm::View<reco::GsfTrack> >              GSFTrack_;

};
#endif
