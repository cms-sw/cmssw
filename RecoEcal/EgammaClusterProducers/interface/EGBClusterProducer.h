//
// Original Author:  David Futyan
//         Created:  Sat Jan 21 17:47:16 CET 2006
// $Id: EGBClusterProducer.h,v 1.4 2006/02/28 18:40:02 tsirig Exp $
//
//
#ifndef RecoEcal_EgammaClusterProducers_EGBClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_EGBClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEcal/EgammaClusterAlgos/interface/IslandClusterAlgo.h"
#include "RecoEcal/EgammaClusterAlgos/interface/HybridClusterAlgo.h"


//


class EGBClusterProducer : public edm::EDProducer 
{
  
 public:
  
  EGBClusterProducer(const edm::ParameterSet&);
  
  ~EGBClusterProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  
  std::string clusterCollection_;
  /** Reconstruction Algorithms*/
  IslandClusterAlgo *island_p;
  HybridClusterAlgo *hybrid_p;
  
  /** Correcctors*/
  //EGBClusPosCalculator* positionCalculator_;
  //bool logWeightedPosition_;
};

#endif
