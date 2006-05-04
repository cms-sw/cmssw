#ifndef RecoEcal_EgammaClusterProducers_IslandClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_IslandClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEcal/EgammaClusterAlgos/interface/IslandClusterAlgo.h"


//


class IslandClusterProducer : public edm::EDProducer 
{
  
  public:

      IslandClusterProducer(const edm::ParameterSet& ps);

      ~IslandClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events
 
      std::string clusterCollection_;
      std::string hitProducer_;
      std::string hitCollection_;

      IslandClusterAlgo * island_p;

      bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0));}
};


#endif
