#ifndef NjettinessAdder_h
#define NjettinessAdder_h

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

class NjettinessAdder : public edm::EDProducer { 
 public:
  explicit NjettinessAdder(const edm::ParameterSet& iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    src_token_(consumes<edm::View<reco::PFJet>>(src_)),
    cone_(iConfig.getParameter<double>("cone"))
      {
	produces<edm::ValueMap<float> >("tau1");
	produces<edm::ValueMap<float> >("tau2");
	produces<edm::ValueMap<float> >("tau3");
      }
    
    virtual ~NjettinessAdder() {}
    
    void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) ;
    float getTau(int num,edm::Ptr<reco::PFJet> object) const;
    
 private:	
    edm::InputTag src_ ;
    edm::EDGetTokenT<edm::View<reco::PFJet>> src_token_;
    double cone_ ;
};

#endif
