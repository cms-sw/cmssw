#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/transform.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

class ReducedESRecHitCollectionProducer : public edm::stream::EDProducer<> {
public:
  ReducedESRecHitCollectionProducer(const edm::ParameterSet& pset);
  ~ReducedESRecHitCollectionProducer() override;
  void beginRun(edm::Run const&, const edm::EventSetup&) final;
  void produce(edm::Event& e, const edm::EventSetup& c) override;
  void collectIds(const ESDetId strip1, const ESDetId strip2, const int& row = 0);

private:
  const EcalPreshowerGeometry* geometry_p;
  std::unique_ptr<CaloSubdetectorTopology> topology_p;

  double scEtThresh_;

  edm::EDGetTokenT<ESRecHitCollection> InputRecHitES_;
  edm::EDGetTokenT<reco::SuperClusterCollection> InputSuperClusterEE_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  std::string OutputLabelES_;
  std::vector<edm::EDGetTokenT<DetIdCollection>> interestingDetIdCollections_;
  std::vector<edm::EDGetTokenT<DetIdCollection>>
      interestingDetIdCollectionsNotToClean_;  //theres a hard coded cut on rec-hit quality which some collections would prefer not to have...

  std::set<DetId> collectedIds_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ReducedESRecHitCollectionProducer);

using namespace edm;
using namespace std;
using namespace reco;

ReducedESRecHitCollectionProducer::ReducedESRecHitCollectionProducer(const edm::ParameterSet& ps)
    : geometry_p(nullptr) {
  scEtThresh_ = ps.getParameter<double>("scEtThreshold");

  InputRecHitES_ = consumes<ESRecHitCollection>(ps.getParameter<edm::InputTag>("EcalRecHitCollectionES"));
  InputSuperClusterEE_ =
      consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("EndcapSuperClusterCollection"));
  caloGeometryToken_ = esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>();

  OutputLabelES_ = ps.getParameter<std::string>("OutputLabel_ES");

  interestingDetIdCollections_ =
      edm::vector_transform(ps.getParameter<std::vector<edm::InputTag>>("interestingDetIds"),
                            [this](edm::InputTag const& tag) { return consumes<DetIdCollection>(tag); });

  interestingDetIdCollectionsNotToClean_ =
      edm::vector_transform(ps.getParameter<std::vector<edm::InputTag>>("interestingDetIdsNotToClean"),
                            [this](edm::InputTag const& tag) { return consumes<DetIdCollection>(tag); });

  produces<EcalRecHitCollection>(OutputLabelES_);
}

ReducedESRecHitCollectionProducer::~ReducedESRecHitCollectionProducer() = default;

void ReducedESRecHitCollectionProducer::beginRun(edm::Run const&, const edm::EventSetup& iSetup) {
  ESHandle<CaloGeometry> geoHandle = iSetup.getHandle(caloGeometryToken_);
  geometry_p =
      dynamic_cast<const EcalPreshowerGeometry*>(geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower));
  if (!geometry_p) {
    edm::LogError("WrongGeometry") << "could not cast the subdet geometry to preshower geometry";
  }

  if (geometry_p)
    topology_p = std::make_unique<EcalPreshowerTopology>();
}

void ReducedESRecHitCollectionProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) {
  edm::Handle<ESRecHitCollection> ESRecHits_;
  e.getByToken(InputRecHitES_, ESRecHits_);

  auto output = std::make_unique<EcalRecHitCollection>();

  edm::Handle<reco::SuperClusterCollection> pEndcapSuperClusters;
  e.getByToken(InputSuperClusterEE_, pEndcapSuperClusters);
  {
    const reco::SuperClusterCollection* eeSuperClusters = pEndcapSuperClusters.product();

    for (reco::SuperClusterCollection::const_iterator isc = eeSuperClusters->begin(); isc != eeSuperClusters->end();
         ++isc) {
      if (isc->energy() < scEtThresh_)
        continue;
      if (fabs(isc->eta()) < 1.65 || fabs(isc->eta()) > 2.6)
        continue;
      //cout<<"SC energy : "<<isc->energy()<<" "<<isc->eta()<<endl;

      //Int_t nBC = 0;
      reco::CaloCluster_iterator ibc = isc->clustersBegin();
      for (; ibc != isc->clustersEnd(); ++ibc) {
        //cout<<"BC : "<<nBC<<endl;

        const GlobalPoint point((*ibc)->x(), (*ibc)->y(), (*ibc)->z());

        ESDetId esId1 = geometry_p->getClosestCellInPlane(point, 1);
        ESDetId esId2 = geometry_p->getClosestCellInPlane(point, 2);

        collectIds(esId1, esId2, 0);
        collectIds(esId1, esId2, 1);
        collectIds(esId1, esId2, -1);

        //nBC++;
      }
    }
  }

  edm::Handle<DetIdCollection> detId;
  for (unsigned int t = 0; t < interestingDetIdCollections_.size(); ++t) {
    e.getByToken(interestingDetIdCollections_[t], detId);
    if (!detId.isValid()) {
      Labels labels;
      labelsForToken(interestingDetIdCollections_[t], labels);
      edm::LogError("MissingInput") << "no reason to skip detid from : (" << labels.module << ", "
                                    << labels.productInstance << ", " << labels.process << ")" << std::endl;
      continue;
    }
    collectedIds_.insert(detId->begin(), detId->end());
  }

  //screw it, cant think of a better solution, not the best but lets run over all the rec hits, remove the ones failing cleaning
  //and then merge in the collection not to be cleaned
  //mainly as I suspect its more efficient to find an object in the DetIdSet rather than the rec-hit in the rec-hit collecition
  //with only a det id
  //if its a CPU issues then revisit
  for (const auto& hit : *ESRecHits_) {
    if (hit.recoFlag() == 1 || hit.recoFlag() == 14 ||
        (hit.recoFlag() <= 10 && hit.recoFlag() >= 5)) {  //right we might need to erase it from the collection
      auto idIt = collectedIds_.find(hit.id());
      if (idIt != collectedIds_.end())
        collectedIds_.erase(idIt);
    }
  }

  for (const auto& token : interestingDetIdCollectionsNotToClean_) {
    e.getByToken(token, detId);
    if (!detId.isValid()) {  //meh might as well keep the warning
      Labels labels;
      labelsForToken(token, labels);
      edm::LogError("MissingInput") << "no reason to skip detid from : (" << labels.module << ", "
                                    << labels.productInstance << ", " << labels.process << ")" << std::endl;
      continue;
    }
    collectedIds_.insert(detId->begin(), detId->end());
  }

  output->reserve(collectedIds_.size());
  EcalRecHitCollection::const_iterator it;
  for (it = ESRecHits_->begin(); it != ESRecHits_->end(); ++it) {
    if (collectedIds_.find(it->id()) != collectedIds_.end()) {
      output->push_back(*it);
    }
  }
  collectedIds_.clear();

  e.put(std::move(output), OutputLabelES_);
}

void ReducedESRecHitCollectionProducer::collectIds(const ESDetId esDetId1, const ESDetId esDetId2, const int& row) {
  //cout<<row<<endl;

  map<DetId, const EcalRecHit*>::iterator it;
  map<DetId, int>::iterator itu;
  ESDetId next;
  ESDetId strip1;
  ESDetId strip2;

  strip1 = esDetId1;
  strip2 = esDetId2;

  EcalPreshowerNavigator theESNav1(strip1, topology_p.get());
  theESNav1.setHome(strip1);

  EcalPreshowerNavigator theESNav2(strip2, topology_p.get());
  theESNav2.setHome(strip2);

  if (row == 1) {
    if (strip1 != ESDetId(0))
      strip1 = theESNav1.north();
    if (strip2 != ESDetId(0))
      strip2 = theESNav2.east();
  } else if (row == -1) {
    if (strip1 != ESDetId(0))
      strip1 = theESNav1.south();
    if (strip2 != ESDetId(0))
      strip2 = theESNav2.west();
  }

  // Plane 1
  if (strip1 == ESDetId(0)) {
  } else {
    collectedIds_.insert(strip1);
    //cout<<"center : "<<strip1<<endl;
    // east road
    for (int i = 0; i < 15; ++i) {
      next = theESNav1.east();
      //cout<<"east : "<<i<<" "<<next<<endl;
      if (next != ESDetId(0)) {
        collectedIds_.insert(next);
      } else {
        break;
      }
    }

    // west road
    theESNav1.setHome(strip1);
    theESNav1.home();
    for (int i = 0; i < 15; ++i) {
      next = theESNav1.west();
      //cout<<"west : "<<i<<" "<<next<<endl;
      if (next != ESDetId(0)) {
        collectedIds_.insert(next);
      } else {
        break;
      }
    }
  }

  if (strip2 == ESDetId(0)) {
  } else {
    collectedIds_.insert(strip2);
    //cout<<"center : "<<strip2<<endl;
    // north road
    for (int i = 0; i < 15; ++i) {
      next = theESNav2.north();
      //cout<<"north : "<<i<<" "<<next<<endl;
      if (next != ESDetId(0)) {
        collectedIds_.insert(next);
      } else {
        break;
      }
    }

    // south road
    theESNav2.setHome(strip2);
    theESNav2.home();
    for (int i = 0; i < 15; ++i) {
      next = theESNav2.south();
      //cout<<"south : "<<i<<" "<<next<<endl;
      if (next != ESDetId(0)) {
        collectedIds_.insert(next);
      } else {
        break;
      }
    }
  }
}
