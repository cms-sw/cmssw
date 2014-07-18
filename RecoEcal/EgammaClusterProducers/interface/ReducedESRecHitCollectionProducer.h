#ifndef _ReducedESRecHitCollectionProducer_H
#define _ReducedESRecHitCollectionProducer_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>

class EcalPreshowerGeometry;
class CaloSubdetectorTopology;
class ReducedESRecHitCollectionProducer : public edm::stream::EDProducer<> {

 public :

  ReducedESRecHitCollectionProducer(const edm::ParameterSet& pset);
  virtual ~ReducedESRecHitCollectionProducer();
  virtual void beginRun (edm::Run const&, const edm::EventSetup&) override final;
  void produce(edm::Event & e, const edm::EventSetup& c);
  void collectIds(const ESDetId strip1, const ESDetId strip2, const int & row=0);
  
 private :

  const EcalPreshowerGeometry *geometry_p;
  CaloSubdetectorTopology *topology_p;

  double scEtThresh_;

  edm::EDGetTokenT<ESRecHitCollection>           InputRecHitES_;  
  edm::EDGetTokenT<reco::SuperClusterCollection> InputSuperClusterEE_;
  std::string OutputLabelES_;
  std::vector<edm::EDGetTokenT<DetIdCollection>> interestingDetIdCollections_;

  std::set<DetId> collectedIds_;
  
};

#endif


