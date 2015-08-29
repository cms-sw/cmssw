#ifndef ECFAdder_h
#define ECFAdder_h

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "fastjet/contrib/EnergyCorrelator.hh"


class ECFAdder : public edm::EDProducer { 
 public:
    explicit ECFAdder(const edm::ParameterSet& iConfig);
    
    virtual ~ECFAdder() {}
    
    void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) ;
    float getECF(unsigned num, const edm::Ptr<reco::Jet> & object) const;
    
 private:	
    edm::InputTag                          src_;
    edm::EDGetTokenT<edm::View<reco::Jet>> src_token_;
    std::vector<unsigned>                  Njets_;
    double                                 beta_ ;

    std::vector<std::auto_ptr<fastjet::contrib::EnergyCorrelator> >  routine_; 
};

#endif
