#ifndef RecoEcal_EgammaClusterProducers_HybridClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_HybridClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaClusterAlgos/interface/HybridClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

//


class HybridClusterProducer : public edm::stream::EDProducer<>
{
  
  public:

      HybridClusterProducer(const edm::ParameterSet& ps);

      ~HybridClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
      int nEvt_;         // internal counter of events
 
      std::string basicclusterCollection_;
      std::string superclusterCollection_;
      
      edm::EDGetTokenT<EcalRecHitCollection> hitsToken_;
 

      HybridClusterAlgo * hybrid_p; // clustering algorithm
      PositionCalc posCalculator_; // position calculation algorithm


};


#endif


