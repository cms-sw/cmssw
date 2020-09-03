// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"

// Geometry
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/Multi5x5ClusterProducer.h"

Multi5x5ClusterProducer::Multi5x5ClusterProducer(const edm::ParameterSet& ps) {
  // Parameters to identify the hit collections
  barrelHitToken_ = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("barrelHitTag"));

  endcapHitToken_ = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("endcapHitTag"));

  //EventSetup Token for CaloGeometry
  caloGeometryToken_ = esConsumes<CaloGeometry, CaloGeometryRecord>();

  // should cluster algo be run in barrel and endcap?
  doEndcap_ = ps.getParameter<bool>("doEndcap");
  doBarrel_ = ps.getParameter<bool>("doBarrel");

  // The names of the produced cluster collections
  barrelClusterCollection_ = ps.getParameter<std::string>("barrelClusterCollection");
  endcapClusterCollection_ = ps.getParameter<std::string>("endcapClusterCollection");

  // Island algorithm parameters
  double barrelSeedThreshold = ps.getParameter<double>("IslandBarrelSeedThr");
  double endcapSeedThreshold = ps.getParameter<double>("IslandEndcapSeedThr");

  const std::vector<std::string> flagnames = ps.getParameter<std::vector<std::string> >("RecHitFlagToBeExcluded");

  const std::vector<int> v_chstatus = StringToEnumValue<EcalRecHit::Flags>(flagnames);

  // Parameters for the position calculation:
  edm::ParameterSet posCalcParameters = ps.getParameter<edm::ParameterSet>("posCalcParameters");
  posCalculator_ = PositionCalc(posCalcParameters);

  // Produces a collection of barrel and a collection of endcap clusters
  produces<reco::BasicClusterCollection>(endcapClusterCollection_);
  produces<reco::BasicClusterCollection>(barrelClusterCollection_);

  bool reassignSeedCrysToClusterItSeeds = false;
  if (ps.exists("reassignSeedCrysToClusterItSeeds"))
    reassignSeedCrysToClusterItSeeds = ps.getParameter<bool>("reassignSeedCrysToClusterItSeeds");

  island_p = new Multi5x5ClusterAlgo(
      barrelSeedThreshold, endcapSeedThreshold, v_chstatus, posCalculator_, reassignSeedCrysToClusterItSeeds);

  nEvt_ = 0;
}

Multi5x5ClusterProducer::~Multi5x5ClusterProducer() { delete island_p; }

void Multi5x5ClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  if (doEndcap_) {
    clusterizeECALPart(evt, es, endcapHitToken_, endcapClusterCollection_, reco::CaloID::DET_ECAL_ENDCAP);
  }
  if (doBarrel_) {
    clusterizeECALPart(evt, es, barrelHitToken_, barrelClusterCollection_, reco::CaloID::DET_ECAL_BARREL);
  }

  nEvt_++;
}

const EcalRecHitCollection* Multi5x5ClusterProducer::getCollection(
    edm::Event& evt, const edm::EDGetTokenT<EcalRecHitCollection>& token) {
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByToken(token, rhcHandle);
  return rhcHandle.product();
}

void Multi5x5ClusterProducer::clusterizeECALPart(edm::Event& evt,
                                                 const edm::EventSetup& es,
                                                 const edm::EDGetTokenT<EcalRecHitCollection>& token,
                                                 const std::string& clusterCollection,
                                                 const reco::CaloID::Detectors detector) {
  // get the hit collection from the event:
  const EcalRecHitCollection* hitCollection_p = getCollection(evt, token);

  // get the geometry and topology from the event setup:
  edm::ESHandle<CaloGeometry> geoHandle = es.getHandle(caloGeometryToken_);

  const CaloSubdetectorGeometry* geometry_p;
  std::unique_ptr<CaloSubdetectorTopology> topology_p;

  if (detector == reco::CaloID::DET_ECAL_BARREL) {
    geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    topology_p = std::make_unique<EcalBarrelTopology>(*geoHandle);
  } else {
    geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    topology_p = std::make_unique<EcalEndcapTopology>(*geoHandle);
  }

  const CaloSubdetectorGeometry* geometryES_p;
  geometryES_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

  // Run the clusterization algorithm:
  reco::BasicClusterCollection clusters;
  clusters = island_p->makeClusters(hitCollection_p, geometry_p, topology_p.get(), geometryES_p, detector);

  // create a unique_ptr to a BasicClusterCollection, copy the barrel clusters into it and put in the Event:
  auto clusters_p = std::make_unique<reco::BasicClusterCollection>();
  clusters_p->assign(clusters.begin(), clusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle;
  if (detector == reco::CaloID::DET_ECAL_BARREL)
    bccHandle = evt.put(std::move(clusters_p), barrelClusterCollection_);
  else
    bccHandle = evt.put(std::move(clusters_p), endcapClusterCollection_);
}
