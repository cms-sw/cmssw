#ifndef RECOMET_METPRODUCERS_PARTICLEFLOWFORCHARGEDMETPRODUCER_H
#define RECOMET_METPRODUCERS_PARTICLEFLOWFORCHARGEDMETPRODUCER_H

/*
  Producer of collection of charged PF candidates beloning to the main PV
  Author: Marco Zanetti, MIT
*/  

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/VertexReco/interface/VertexFwd.h>
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h>

namespace reco
{
  class ParticleFlowForChargedMETProducer : public edm::stream::EDProducer<> {
    
  public:
    explicit ParticleFlowForChargedMETProducer(const edm::ParameterSet&);
    ~ParticleFlowForChargedMETProducer() override;
    
  private:
    
    void produce(edm::Event&, const edm::EventSetup&) override;
    
    edm::InputTag pfCollectionLabel;
    edm::InputTag pvCollectionLabel;

    edm::EDGetTokenT<VertexCollection> pvCollectionToken;
    edm::EDGetTokenT<PFCandidateCollection> pfCandidatesToken;

    double dzCut;
    double neutralEtThreshold;
  };
}

#endif
  
