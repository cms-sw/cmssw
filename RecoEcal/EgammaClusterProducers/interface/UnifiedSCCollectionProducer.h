#ifndef RecoEcal_EgammaClusterProducers_UnifiedSCCollectionProducer_h_
#define RecoEcal_EgammaClusterProducers_UnifiedSCCollectionProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

class UnifiedSCCollectionProducer : public edm::stream::EDProducer<> 
{
  
  public:

      UnifiedSCCollectionProducer(const edm::ParameterSet& ps);

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
      std::string  bcCollectionUncleanOnly_;
      std::string  scCollectionUncleanOnly_;



};


#endif


