#ifndef RecoEcal_EgammaClusterProducers_TestClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_TestClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEcal/EgammaClusterAlgos/interface/IslandClusterAlgo.h"


//


class TestClusterProducer : public edm::EDProducer 
{
  
  public:

      TestClusterProducer(const edm::ParameterSet& ps);

      ~TestClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events
 
      std::string clusterCollection_;
      IslandClusterAlgo * island_p;

};


#endif


