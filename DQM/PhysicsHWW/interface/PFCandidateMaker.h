#ifndef PFCANDIDATEMAKER_H
#define PFCANDIDATEMAKER_H

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class PFCandidateMaker {

  public:

    PFCandidateMaker(const edm::ParameterSet&, edm::ConsumesCollector);
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

  private:

    edm::EDGetTokenT<reco::PFCandidateCollection>             PFCandidateCollection_;
    edm::EDGetTokenT<edm::ValueMap<reco::PFCandidatePtr> >    PFElectrons_;
    edm::EDGetTokenT<reco::TrackCollection>                   TrackCollection_;
    edm::EDGetTokenT<reco::VertexCollection>                  thePVCollection_;

};

#endif

