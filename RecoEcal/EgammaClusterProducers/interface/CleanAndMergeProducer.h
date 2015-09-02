#ifndef RecoEcal_EgammaClusterProducers_CleanAndMergeProducer_h_
#define RecoEcal_EgammaClusterProducers_CleanAndMergeProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"


class CleanAndMergeProducer : public edm::stream::EDProducer<>
{
  
  public:

      CleanAndMergeProducer(const edm::ParameterSet& ps);

      ~CleanAndMergeProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
      
  private:
      

      edm::EDGetTokenT<reco::SuperClusterCollection> cleanScToken_;
      edm::EDGetTokenT<reco::SuperClusterCollection> uncleanScToken_;
     
      // the names of the products to be produced:
      std::string  bcCollection_;     
      std::string  scCollection_;     
      std::string  refScCollection_;  


};


#endif


