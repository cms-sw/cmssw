#ifndef RecoEcal_EgammaClusterProducers_HybridClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_HybridClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEcal/EgammaClusterAlgos/interface/HybridClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

//


class HybridClusterProducer : public edm::EDProducer 
{
  
  public:

      HybridClusterProducer(const edm::ParameterSet& ps);

      ~HybridClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
      int nEvt_;         // internal counter of events
 
      std::string basicclusterCollection_;
      std::string superclusterCollection_;
      std::string hitproducer_;
      std::string hitcollection_;


      HybridClusterAlgo * hybrid_p; // clustering algorithm
      PositionCalc posCalculator_; // position calculation algorithm


};


#endif


