#ifndef RecoEcal_EgammaClusterProducers_UncleanSCRecoveryProducer_h_
#define RecoEcal_EgammaClusterProducers_UncleanSCRecoveryProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"



class UncleanSCRecoveryProducer : public edm::global::EDProducer<> 
{
  
  public:

      UncleanSCRecoveryProducer(const edm::ParameterSet& ps);

      virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
      
  private:
      // the clean collection      
      const edm::EDGetTokenT<reco::BasicClusterCollection>  cleanBcCollection_; 
      const edm::EDGetTokenT<reco::SuperClusterCollection>  cleanScCollection_; 
      // the uncleaned collection
      const edm::EDGetTokenT<reco::BasicClusterCollection>  uncleanBcCollection_;
      const edm::EDGetTokenT<reco::SuperClusterCollection>  uncleanScCollection_;
      // the names of the products to be produced:
      const std::string  bcCollection_;     
      const std::string  scCollection_;     


};


#endif


