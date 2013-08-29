#ifndef RecoEcal_EgammaClusterProducers_UncleanSCRecoveryProducer_h_
#define RecoEcal_EgammaClusterProducers_UncleanSCRecoveryProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"


// $Id$

class UncleanSCRecoveryProducer : public edm::EDProducer 
{
  
  public:

      UncleanSCRecoveryProducer(const edm::ParameterSet& ps);

      virtual void produce(edm::Event&, const edm::EventSetup&);
      
  private:
      // the clean collection      
      edm::EDGetTokenT<reco::BasicClusterCollection>  cleanBcCollection_; 
      edm::EDGetTokenT<reco::SuperClusterCollection>  cleanScCollection_; 
      // the uncleaned collection
      edm::EDGetTokenT<reco::BasicClusterCollection>  uncleanBcCollection_;
      edm::EDGetTokenT<reco::SuperClusterCollection>  uncleanScCollection_;
      // the names of the products to be produced:
      std::string  bcCollection_;     
      std::string  scCollection_;     


};


#endif


