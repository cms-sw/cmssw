#ifndef NjettinessAdder_h
#define NjettinessAdder_h

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/ValueMap.h"

class NjettinessAdder : public edm::EDProducer { 
 public:
    explicit NjettinessAdder(const edm::ParameterSet& iConfig) :
      src_(iConfig.getParameter<edm::InputTag>("src")),
      src_token_(consumes<edm::View<reco::Jet>>(src_)),
      cone_(iConfig.getParameter<double>("cone")),
      Njets_(iConfig.getParameter<std::vector<unsigned> >("Njets"))
    {
      for ( std::vector<unsigned>::const_iterator n = Njets_.begin(); n != Njets_.end(); ++n )
      {
        std::ostringstream tauN_str;
        tauN_str << "tau" << *n;

        produces<edm::ValueMap<float> >(tauN_str.str().c_str());
      }
    }
    
    virtual ~NjettinessAdder() {}
    
    void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) ;
    float getTau(unsigned num, const edm::Ptr<reco::Jet> & object) const;
    
 private:	
    const edm::InputTag                          src_;
    const edm::EDGetTokenT<edm::View<reco::Jet>> src_token_;
    const double                                 cone_ ;
    const std::vector<unsigned>                  Njets_;
};

#endif
