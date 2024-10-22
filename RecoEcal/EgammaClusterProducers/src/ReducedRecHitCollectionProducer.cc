// -*- C++ -*-
//
// Package:    ReducedRecHitCollectionProducer
// Class:      ReducedRecHitCollectionProducer
//
/**\class ReducedRecHitCollectionProducer ReducedRecHitCollectionProducer.cc Calibration/EcalAlCaRecoProducers/src/ReducedRecHitCollectionProducer.cc

Original author: Paolo Meridiani PH/CMG
 
Implementation:
 <Notes on implementation>
*/

#include <iostream>
#include <memory>

#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

class ReducedRecHitCollectionProducer : public edm::stream::EDProducer<> {
public:
  //! ctor
  explicit ReducedRecHitCollectionProducer(const edm::ParameterSet&);
  ~ReducedRecHitCollectionProducer() override;
  //! producer
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  edm::EDGetTokenT<EcalRecHitCollection> recHitsToken_;
  std::vector<edm::EDGetTokenT<DetIdCollection>> interestingDetIdCollections_;
  std::string reducedHitsCollection_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ReducedRecHitCollectionProducer);

ReducedRecHitCollectionProducer::ReducedRecHitCollectionProducer(const edm::ParameterSet& iConfig) {
  recHitsToken_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsLabel"));

  interestingDetIdCollections_ =
      edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag>>("interestingDetIdCollections"),
                            [this](edm::InputTag const& tag) { return consumes<DetIdCollection>(tag); });

  reducedHitsCollection_ = iConfig.getParameter<std::string>("reducedHitsCollection");

  //register your products
  produces<EcalRecHitCollection>(reducedHitsCollection_);
}

ReducedRecHitCollectionProducer::~ReducedRecHitCollectionProducer() {}

// ------------ method called to produce the data  ------------
void ReducedRecHitCollectionProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  if (interestingDetIdCollections_.empty()) {
    edm::LogError("ReducedRecHitCollectionProducer") << "VInputTag collections empty";
    return;
  }

  Handle<DetIdCollection> detIds;
  iEvent.getByToken(interestingDetIdCollections_[0], detIds);
  std::vector<DetId> xtalsToStore((*detIds).size());
  std::copy((*detIds).begin(), (*detIds).end(), xtalsToStore.begin());

  //Merging DetIds from different collections
  for (unsigned int t = 1; t < interestingDetIdCollections_.size(); ++t) {
    Handle<DetIdCollection> detId;
    iEvent.getByToken(interestingDetIdCollections_[t], detId);
    if (!detId.isValid()) {
      Labels labels;
      labelsForToken(interestingDetIdCollections_[t], labels);
      edm::LogError("MissingInput") << "no reason to skip detid from : (" << labels.module << ", "
                                    << labels.productInstance << ", " << labels.process << ")" << std::endl;
      continue;
    }

    for (unsigned int ii = 0; ii < (*detId).size(); ii++) {
      if (std::find(xtalsToStore.begin(), xtalsToStore.end(), (*detId)[ii]) == xtalsToStore.end())
        xtalsToStore.push_back((*detId)[ii]);
    }
  }

  Handle<EcalRecHitCollection> recHitsHandle;
  iEvent.getByToken(recHitsToken_, recHitsHandle);
  if (!recHitsHandle.isValid()) {
    edm::LogError("ReducedRecHitCollectionProducer") << "RecHit collection not found";
    return;
  }

  //Create empty output collections
  auto miniRecHitCollection = std::make_unique<EcalRecHitCollection>();

  for (unsigned int iCry = 0; iCry < xtalsToStore.size(); iCry++) {
    EcalRecHitCollection::const_iterator iRecHit = recHitsHandle->find(xtalsToStore[iCry]);
    if ((iRecHit != recHitsHandle->end()) &&
        (miniRecHitCollection->find(xtalsToStore[iCry]) == miniRecHitCollection->end()))
      miniRecHitCollection->push_back(*iRecHit);
  }

  std::sort(xtalsToStore.begin(), xtalsToStore.end());
  std::unique(xtalsToStore.begin(), xtalsToStore.end());

  //   std::cout << "New Collection " << reducedHitsCollection_ << " size is " << miniRecHitCollection->size() << " original is " << recHitsHandle->size() << std::endl;
  iEvent.put(std::move(miniRecHitCollection), reducedHitsCollection_);
}
