#ifndef _ReducedESRecHitCollectionProducer_H
#define _ReducedESRecHitCollectionProducer_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>

class EcalPreshowerGeometry;
class CaloSubdetectorTopology;
class ReducedESRecHitCollectionProducer : public edm::EDProducer {

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

  edm::InputTag InputRecHitES_;  
  edm::InputTag InputSpuerClusterEE_;
  std::string OutputLabelES_;
  std::vector<edm::InputTag> interestingDetIdCollections_;

  std::set<DetId> collectedIds_;
  
};

#endif


