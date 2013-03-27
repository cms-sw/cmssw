#ifndef RECOMET_METPRODUCERS_PARTICLEFLOWFORCHARGEDMETPRODUCER_H
#define RECOMET_METPRODUCERS_PARTICLEFLOWFORCHARGEDMETPRODUCER_H

/*
  Producer of collection of charged PF candidates beloning to the main PV
  Author: Marco Zanetti, MIT
*/  

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco
{
  class ParticleFlowForChargedMETProducer : public edm::EDProducer {
    
  public:
    explicit ParticleFlowForChargedMETProducer(const edm::ParameterSet&);
    ~ParticleFlowForChargedMETProducer();
    
  private:
    
    virtual void produce(edm::Event&, const edm::EventSetup&);
    
    edm::InputTag pfCollectionLabel;
    edm::InputTag pvCollectionLabel;

    double dzCut;
    double neutralEtThreshold;
  };
}

#endif
  
