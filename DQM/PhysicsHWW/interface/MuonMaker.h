#ifndef MUONMAKER_H
#define MUONMAKER_H

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/MuonReco/interface/MuonShower.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class MuonMaker {

  public:

    MuonMaker(const edm::ParameterSet&, edm::ConsumesCollector);
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

  private:

    edm::EDGetTokenT<edm::View<reco::Muon> >                  Muon_;
    edm::EDGetTokenT<edm::ValueMap<reco::MuonShower> >        MuonShower_;
    edm::EDGetTokenT<reco::VertexCollection>                  thePVCollection_;
    edm::EDGetTokenT<reco::PFCandidateCollection>             PFCandidateCollection_;
    edm::EDGetTokenT<reco::BeamSpot>                          BeamSpot_;
    edm::EDGetTokenT<reco::MuonCollection>                    MuonCollection_;

};

#endif
