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
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>

using namespace edm;
using namespace std;
using namespace reco;

class ReducedESRecHitCollectionProducer : public edm::EDProducer {

 public :

  ReducedESRecHitCollectionProducer(const edm::ParameterSet& pset);
  virtual ~ReducedESRecHitCollectionProducer();
  void produce(edm::Event & e, const edm::EventSetup& c);
  void beginJob(void);
  void endJob(void);
  EcalRecHitCollection getESHits(double X, double Y, double Z, const CaloSubdetectorGeometry*& geometry_p, CaloSubdetectorTopology *topology_p, int row=0);
  
 private :

  double scEtThresh_;

  edm::InputTag InputRecHitES_;  
  edm::InputTag InputSpuerClusterEE_;
  std::string OutputLabelES_;

  map<DetId, EcalRecHit> rechits_map_;
  map<DetId, int> used_strips_;
  
};

#endif


