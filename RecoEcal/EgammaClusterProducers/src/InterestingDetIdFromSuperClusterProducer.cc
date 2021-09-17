// -*- C++ -*-
//
// Package:    InterestingDetIdFromSuperClusterProducer
// Class:      InterestingDetIdFromSuperClusterProducer
//
/**\class InterestingDetIdFromSuperClusterProducer 
Adapted from InterestingDetIdCollectionProducer by J.Bendavid
 
Make a collection of detids to be kept tipically in a AOD rechit collection

The following classes of "interesting id" are considered

    1.All rechits included in all subclusters, plus in a region around  the seed of each subcluster
      The size of the region is specified by
      minimalEtaSize_, minimalPhiSize_
 
    2. if the severity of the hit is >= severityLevel_
       If severityLevel=0 this class is ignored

    3. Channels next to dead ones,  keepNextToDead_ is true
    4. Channels next to the EB/EE transition if keepNextToBoundary_ is true
*/

#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
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
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalNextToDeadChannelRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include <memory>

class InterestingDetIdFromSuperClusterProducer : public edm::stream::EDProducer<> {
public:
  //! ctor
  explicit InterestingDetIdFromSuperClusterProducer(const edm::ParameterSet&);
  void beginRun(edm::Run const&, const edm::EventSetup&) final;
  //! producer
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  edm::EDGetTokenT<EcalRecHitCollection> recHitsToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> superClustersToken_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopologyToken_;
  edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> severityLevelToken_;
  edm::ESGetToken<EcalNextToDeadChannel, EcalNextToDeadChannelRcd> nextToDeadToken_;
  std::string interestingDetIdCollection_;
  int minimalEtaSize_;
  int minimalPhiSize_;
  const CaloTopology* caloTopology_;

  int severityLevel_;
  const EcalSeverityLevelAlgo* severity_;
  bool keepNextToDead_;
  bool keepNextToBoundary_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(InterestingDetIdFromSuperClusterProducer);

InterestingDetIdFromSuperClusterProducer::InterestingDetIdFromSuperClusterProducer(const edm::ParameterSet& iConfig) {
  recHitsToken_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsLabel"));
  superClustersToken_ =
      consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("superClustersLabel"));
  caloTopologyToken_ = esConsumes<CaloTopology, CaloTopologyRecord, edm::Transition::BeginRun>();
  severityLevelToken_ = esConsumes<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd, edm::Transition::BeginRun>();
  interestingDetIdCollection_ = iConfig.getParameter<std::string>("interestingDetIdCollection");

  minimalEtaSize_ = iConfig.getParameter<int>("etaSize");
  minimalPhiSize_ = iConfig.getParameter<int>("phiSize");
  if (minimalPhiSize_ % 2 == 0 || minimalEtaSize_ % 2 == 0)
    edm::LogError("InterestingDetIdFromSuperClusterProducerError") << "Size of eta/phi should be odd numbers";

  //register your products
  produces<DetIdCollection>(interestingDetIdCollection_);

  severityLevel_ = iConfig.getParameter<int>("severityLevel");
  keepNextToDead_ = iConfig.getParameter<bool>("keepNextToDead");
  keepNextToBoundary_ = iConfig.getParameter<bool>("keepNextToBoundary");
  if (keepNextToDead_) {
    nextToDeadToken_ = esConsumes<EcalNextToDeadChannel, EcalNextToDeadChannelRcd>();
  }
}

void InterestingDetIdFromSuperClusterProducer::beginRun(edm::Run const& run, const edm::EventSetup& iSetup) {
  edm::ESHandle<CaloTopology> theCaloTopology = iSetup.getHandle(caloTopologyToken_);
  caloTopology_ = &(*theCaloTopology);

  edm::ESHandle<EcalSeverityLevelAlgo> sevLv = iSetup.getHandle(severityLevelToken_);
  severity_ = sevLv.product();
}

// ------------ method called to produce the data  ------------
void InterestingDetIdFromSuperClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  // take BasicClusters
  Handle<reco::SuperClusterCollection> pClusters;
  iEvent.getByToken(superClustersToken_, pClusters);

  // take EcalRecHits
  Handle<EcalRecHitCollection> recHitsHandle;
  iEvent.getByToken(recHitsToken_, recHitsHandle);

  //Create empty output collections
  std::vector<DetId> indexToStore;
  indexToStore.reserve(1000);

  reco::SuperClusterCollection::const_iterator sclusIt;

  std::vector<DetId> xtalsToStore;
  xtalsToStore.reserve(50);

  //loop over superclusters
  for (sclusIt = pClusters->begin(); sclusIt != pClusters->end(); sclusIt++) {
    //loop over subclusters
    for (reco::CaloCluster_iterator clusIt = sclusIt->clustersBegin(); clusIt != sclusIt->clustersEnd(); ++clusIt) {
      //PG barrel

      float eMax = 0.;
      DetId eMaxId(0);

      std::vector<std::pair<DetId, float> > clusterDetIds = (*clusIt)->hitsAndFractions();
      std::vector<std::pair<DetId, float> >::iterator posCurrent;

      EcalRecHit testEcalRecHit;

      for (posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++) {
        EcalRecHitCollection::const_iterator itt = recHitsHandle->find((*posCurrent).first);
        if ((!((*posCurrent).first.null())) && (itt != recHitsHandle->end()) && ((*itt).energy() > eMax)) {
          eMax = (*itt).energy();
          eMaxId = (*itt).id();
        }
      }

      if (eMaxId.null())
        continue;

      const CaloSubdetectorTopology* topology = caloTopology_->getSubdetectorTopology(eMaxId.det(), eMaxId.subdetId());

      xtalsToStore = topology->getWindow(eMaxId, minimalEtaSize_, minimalPhiSize_);
      std::vector<std::pair<DetId, float> > xtalsInClus = (*clusIt)->hitsAndFractions();

      for (unsigned int ii = 0; ii < xtalsInClus.size(); ii++) {
        xtalsToStore.push_back(xtalsInClus[ii].first);
      }

      indexToStore.insert(indexToStore.end(), xtalsToStore.begin(), xtalsToStore.end());
    }
  }

  for (EcalRecHitCollection::const_iterator it = recHitsHandle->begin(); it != recHitsHandle->end(); ++it) {
    // also add recHits of dead TT if the corresponding TP is saturated
    if (it->checkFlag(EcalRecHit::kTPSaturated)) {
      indexToStore.push_back(it->id());
    }
    // add hits for severities above a threshold
    if (severityLevel_ >= 0 && severity_->severityLevel(*it) >= severityLevel_) {
      indexToStore.push_back(it->id());
    }
    if (keepNextToDead_) {
      edm::ESHandle<EcalNextToDeadChannel> dch = iSetup.getHandle(nextToDeadToken_);
      // also keep channels next to dead ones
      if (EcalTools::isNextToDead(it->id(), *dch)) {
        indexToStore.push_back(it->id());
      }
    }

    if (keepNextToBoundary_) {
      // keep channels around EB/EE boundary
      if (it->id().subdetId() == EcalBarrel) {
        EBDetId ebid(it->id());
        if (abs(ebid.ieta()) == 85)
          indexToStore.push_back(it->id());
      } else {
        if (EEDetId::isNextToRingBoundary(it->id()))
          indexToStore.push_back(it->id());
      }
    }
  }

  //unify the vector
  std::sort(indexToStore.begin(), indexToStore.end());
  std::unique(indexToStore.begin(), indexToStore.end());

  auto detIdCollection = std::make_unique<DetIdCollection>(indexToStore);

  iEvent.put(std::move(detIdCollection), interestingDetIdCollection_);
}
