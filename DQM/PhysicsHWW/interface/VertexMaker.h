#ifndef VERTEXMAKER_H
#define VERTEXMAKER_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class VertexMaker {

  public:

    VertexMaker(const edm::ParameterSet&, edm::ConsumesCollector);
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

    edm::EDGetTokenT<reco::VertexCollection> thePVCollection_;

};

#endif
