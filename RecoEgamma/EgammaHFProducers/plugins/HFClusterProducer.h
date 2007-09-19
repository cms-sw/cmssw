#ifndef EgammaElectronProducers_HFClusterProducer_h
#define EgammaElectronProducers_HFClusterProducer_h
//Package:    EgammaHFProdcers
// Class  :    HFClusterProducer
// Original Author:  Kevin Klapoetke (minnesota)
//        
// $Id: HFClusterProducer.h,v 1.2 2007/09/19 Kevin Klapoetke
//
#include "RecoEgamma/EgammaHFProducers/interface/HFClusterAlgo.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

class HFClusterProducer : public edm::EDProducer {
public:
  explicit HFClusterProducer(edm::ParameterSet const& conf);
  virtual void produce(edm::Event& e, edm::EventSetup const& iSetup);
private:
 
  HFClusterAlgo algo_;
};
#endif
