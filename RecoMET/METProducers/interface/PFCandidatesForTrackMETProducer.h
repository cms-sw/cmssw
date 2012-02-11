#ifndef RECOMET_METPRODUCERS_PFCANDIDATESFORTRACKMETPRODUCER_H
#define RECOMET_METPRODUCERS_PFCANDIDATESFORTRACKMETPRODUCER_H

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
  class PFCandidatesForTrackMETProducer : public edm::EDProducer {
    
  public:
    explicit PFCandidatesForTrackMETProducer(const edm::ParameterSet&);
    ~PFCandidatesForTrackMETProducer();
    
  private:
    
    virtual void beginJob() ;
    virtual void endJob() ;
    virtual void produce(edm::Event&, const edm::EventSetup&);
    virtual void beginRun(edm::Run&, const edm::EventSetup&);
    virtual void endRun(edm::Run&, const edm::EventSetup&);
    
    edm::InputTag pfCollectionLabel;
    edm::InputTag pvCollectionLabel;

    double dzCut;
    double neutralEtThreshold;
  };
}

#endif
  
