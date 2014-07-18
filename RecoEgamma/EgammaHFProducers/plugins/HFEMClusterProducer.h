#ifndef EgammaElectronProducers_HFEMClusterProducer_h
#define EgammaElectronProducers_HFEMClusterProducer_h
//Package:    EgammaHFProdcers
// Class  :    HFClusterProducer
// Original Author:  Kevin Klapoetke (minnesota)
//        
// $Id: HFClusterProducer.h,v 1.2 2007/09/19 Kevin Klapoetke
//
#include "HFClusterAlgo.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

class HFEMClusterProducer : public edm::stream::EDProducer<> {
public:
  explicit HFEMClusterProducer(edm::ParameterSet const& conf);
  virtual void produce(edm::Event& e, edm::EventSetup const& iSetup);
  virtual void beginRun(edm::Run const &, edm::EventSetup const&) override final { algo_.resetForRun(); }
private:
  edm::EDGetToken hfreco_;
  HFClusterAlgo algo_;
};
#endif
