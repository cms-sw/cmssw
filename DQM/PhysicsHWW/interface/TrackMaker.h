#ifndef TRACKMAKER_H
#define TRACKMAKER_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class TrackMaker {

  public:

    TrackMaker(const edm::ParameterSet&, edm::ConsumesCollector);
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

    edm::EDGetTokenT<edm::View<reco::Track> > TrackCollection_;

};

#endif
