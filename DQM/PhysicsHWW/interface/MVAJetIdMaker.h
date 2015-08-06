#ifndef MVAJETIDMAKER_H
#define MVAJETIDMAKER_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "RecoJets/JetProducers/interface/PileupJetIdAlgo.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

namespace reco {
  class JetCorrector;
}

class MVAJetIdMaker {

  public:

    MVAJetIdMaker(const edm::ParameterSet&, edm::ConsumesCollector &&);
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

  private:

    edm::EDGetTokenT<reco::PFJetCollection>       PFJetCollection_;
    edm::EDGetTokenT<reco::VertexCollection>      thePVCollection_;
    edm::EDGetTokenT<double> theRhoCollection_;
    edm::EDGetTokenT<reco::JetCorrector> jetCorrectorToken_;
    PileupJetIdAlgo  *fPUJetIdAlgo;

};

#endif
