// -*- C++ -*-

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// to access recHits and BasicClusters
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

class ProbClustersFilter : public edm::stream::EDFilter<> {
public:
  explicit ProbClustersFilter(const edm::ParameterSet&);
  ~ProbClustersFilter() override = default;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  int maxDistance_;
  float maxGoodFraction_;
  edm::ParameterSet conf_;
  edm::InputTag barrelClusterCollection_;
  edm::InputTag endcapClusterCollection_;
  edm::InputTag reducedBarrelRecHitCollection_;
  edm::InputTag reducedEndcapRecHitCollection_;

  const edm::EDGetTokenT<reco::SuperClusterCollection> ebSCToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> ebRecHitsToken_;
  const edm::ESGetToken<CaloTopology, CaloTopologyRecord> topologyToken_;
  const edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> severityLevelAlgoToken_;
};

ProbClustersFilter::ProbClustersFilter(const edm::ParameterSet& iConfig)
    : maxDistance_(iConfig.getParameter<int>("maxDistance")),
      maxGoodFraction_(iConfig.getParameter<double>("maxGoodFraction")),
      conf_(iConfig),
      barrelClusterCollection_(iConfig.getParameter<edm::InputTag>("barrelClusterCollection")),
      endcapClusterCollection_(iConfig.getParameter<edm::InputTag>("endcapClusterCollection")),
      reducedBarrelRecHitCollection_(iConfig.getParameter<edm::InputTag>("reducedBarrelRecHitCollection")),
      reducedEndcapRecHitCollection_(iConfig.getParameter<edm::InputTag>("reducedEndcapRecHitCollection")),
      ebSCToken_(consumes<reco::SuperClusterCollection>(barrelClusterCollection_)),
      ebRecHitsToken_(consumes<EcalRecHitCollection>(reducedBarrelRecHitCollection_)),
      topologyToken_(esConsumes()),
      severityLevelAlgoToken_(esConsumes()) {}

bool ProbClustersFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  int problematicClusters = 0;

  edm::Handle<reco::SuperClusterCollection> pEBClusters;
  iEvent.getByToken(ebSCToken_, pEBClusters);
  const reco::SuperClusterCollection* ebClusters = pEBClusters.product();

  edm::Handle<EcalRecHitCollection> pEBRecHits;
  iEvent.getByToken(ebRecHitsToken_, pEBRecHits);
  const EcalRecHitCollection* ebRecHits = pEBRecHits.product();

  const auto& topology = iSetup.getData(topologyToken_);
  const auto& sevLv = iSetup.getData(severityLevelAlgoToken_);

  for (reco::SuperClusterCollection::const_iterator it = ebClusters->begin(); it != ebClusters->end(); ++it) {
    float goodFraction = EcalClusterSeverityLevelAlgo::goodFraction(*it, *ebRecHits, sevLv);
    std::pair<int, int> distance =
        EcalClusterSeverityLevelAlgo::etaphiDistanceClosestProblematic(*it, *ebRecHits, &topology, sevLv);
    if (distance.first == -1 && distance.second == -1) {
      distance.first = 999;
      distance.second = 999;
    }
    if (goodFraction >= maxGoodFraction_ &&
        sqrt(distance.first * distance.first + distance.second * distance.second) >= maxDistance_)
      continue;
    ++problematicClusters;
  }

  return problematicClusters;
}

DEFINE_FWK_MODULE(ProbClustersFilter);
