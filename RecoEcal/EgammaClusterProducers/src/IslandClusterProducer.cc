#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEcal/EgammaClusterAlgos/interface/IslandClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

#include <ctime>
#include <iostream>
#include <memory>
#include <vector>

class IslandClusterProducer : public edm::stream::EDProducer<> {
public:
  IslandClusterProducer(const edm::ParameterSet& ps);

  ~IslandClusterProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  int nMaxPrintout_;  // max # of printouts
  int nEvt_;          // internal counter of events

  IslandClusterAlgo::VerbosityLevel verbosity;

  edm::EDGetTokenT<EcalRecHitCollection> barrelRecHits_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapRecHits_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;

  std::string barrelClusterCollection_;
  std::string endcapClusterCollection_;

  std::string clustershapecollectionEB_;
  std::string clustershapecollectionEE_;

  //BasicClusterShape AssociationMap
  std::string barrelClusterShapeAssociation_;
  std::string endcapClusterShapeAssociation_;

  PositionCalc posCalculator_;  // position calculation algorithm
  ClusterShapeAlgo shapeAlgo_;  // cluster shape algorithm
  IslandClusterAlgo* island_p;

  bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0)); }

  const EcalRecHitCollection* getCollection(edm::Event& evt, const edm::EDGetTokenT<EcalRecHitCollection>& token);

  void clusterizeECALPart(edm::Event& evt,
                          const edm::EventSetup& es,
                          const edm::EDGetTokenT<EcalRecHitCollection>& token,
                          const std::string& clusterCollection,
                          const std::string& clusterShapeAssociation,
                          const IslandClusterAlgo::EcalPart& ecalPart);

  void outputValidationInfo(reco::CaloClusterPtrVector& clusterPtrVector);
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(IslandClusterProducer);

IslandClusterProducer::IslandClusterProducer(const edm::ParameterSet& ps) {
  // The verbosity level
  std::string verbosityString = ps.getParameter<std::string>("VerbosityLevel");
  if (verbosityString == "DEBUG")
    verbosity = IslandClusterAlgo::pDEBUG;
  else if (verbosityString == "WARNING")
    verbosity = IslandClusterAlgo::pWARNING;
  else if (verbosityString == "INFO")
    verbosity = IslandClusterAlgo::pINFO;
  else
    verbosity = IslandClusterAlgo::pERROR;

  // Parameters to identify the hit collections
  barrelRecHits_ = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("barrelHits"));
  endcapRecHits_ = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("endcapHits"));

  //EventSetup Token for CaloGeometry
  caloGeometryToken_ = esConsumes<CaloGeometry, CaloGeometryRecord>();

  // The names of the produced cluster collections
  barrelClusterCollection_ = ps.getParameter<std::string>("barrelClusterCollection");
  endcapClusterCollection_ = ps.getParameter<std::string>("endcapClusterCollection");

  // Island algorithm parameters
  double barrelSeedThreshold = ps.getParameter<double>("IslandBarrelSeedThr");
  double endcapSeedThreshold = ps.getParameter<double>("IslandEndcapSeedThr");

  // Parameters for the position calculation:
  edm::ParameterSet posCalcParameters = ps.getParameter<edm::ParameterSet>("posCalcParameters");
  posCalculator_ = PositionCalc(posCalcParameters);
  shapeAlgo_ = ClusterShapeAlgo(posCalcParameters);

  clustershapecollectionEB_ = ps.getParameter<std::string>("clustershapecollectionEB");
  clustershapecollectionEE_ = ps.getParameter<std::string>("clustershapecollectionEE");

  //AssociationMap
  barrelClusterShapeAssociation_ = ps.getParameter<std::string>("barrelShapeAssociation");
  endcapClusterShapeAssociation_ = ps.getParameter<std::string>("endcapShapeAssociation");

  const std::vector<std::string> seedflagnamesEB =
      ps.getParameter<std::vector<std::string>>("SeedRecHitFlagToBeExcludedEB");
  const std::vector<int> seedflagsexclEB = StringToEnumValue<EcalRecHit::Flags>(seedflagnamesEB);

  const std::vector<std::string> seedflagnamesEE =
      ps.getParameter<std::vector<std::string>>("SeedRecHitFlagToBeExcludedEE");
  const std::vector<int> seedflagsexclEE = StringToEnumValue<EcalRecHit::Flags>(seedflagnamesEE);

  const std::vector<std::string> flagnamesEB = ps.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEB");
  const std::vector<int> flagsexclEB = StringToEnumValue<EcalRecHit::Flags>(flagnamesEB);

  const std::vector<std::string> flagnamesEE = ps.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEE");
  const std::vector<int> flagsexclEE = StringToEnumValue<EcalRecHit::Flags>(flagnamesEE);

  // Produces a collection of barrel and a collection of endcap clusters

  produces<reco::ClusterShapeCollection>(clustershapecollectionEE_);
  produces<reco::BasicClusterCollection>(endcapClusterCollection_);
  produces<reco::ClusterShapeCollection>(clustershapecollectionEB_);
  produces<reco::BasicClusterCollection>(barrelClusterCollection_);
  produces<reco::BasicClusterShapeAssociationCollection>(barrelClusterShapeAssociation_);
  produces<reco::BasicClusterShapeAssociationCollection>(endcapClusterShapeAssociation_);

  island_p = new IslandClusterAlgo(barrelSeedThreshold,
                                   endcapSeedThreshold,
                                   posCalculator_,
                                   seedflagsexclEB,
                                   seedflagsexclEE,
                                   flagsexclEB,
                                   flagsexclEE,
                                   verbosity);

  nEvt_ = 0;
}

IslandClusterProducer::~IslandClusterProducer() { delete island_p; }

void IslandClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("VerbosityLevel", "ERROR");
  desc.add<edm::InputTag>("barrelHits", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("endcapHits", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<std::string>("barrelClusterCollection", "islandBarrelBasicClusters");
  desc.add<std::string>("endcapClusterCollection", "islandEndcapBasicClusters");
  desc.add<double>("IslandBarrelSeedThr", 0.5);
  desc.add<double>("IslandEndcapSeedThr", 0.18);

  edm::ParameterSetDescription posCalcParameters;
  posCalcParameters.add<bool>("LogWeighted", true);
  posCalcParameters.add<double>("T0_barl", 7.4);
  posCalcParameters.add<double>("T0_endc", 3.1);
  posCalcParameters.add<double>("T0_endcPresh", 1.2);
  posCalcParameters.add<double>("W0", 4.2);
  posCalcParameters.add<double>("X0", 0.89);
  desc.add<edm::ParameterSetDescription>("posCalcParameters", posCalcParameters);

  desc.add<std::string>("clustershapecollectionEE", "islandEndcapShape");
  desc.add<std::string>("clustershapecollectionEB", "islandBarrelShape");
  desc.add<std::string>("barrelShapeAssociation", "islandBarrelShapeAssoc");
  desc.add<std::string>("endcapShapeAssociation", "islandEndcapShapeAssoc");
  desc.add<std::vector<std::string>>("SeedRecHitFlagToBeExcludedEB", {});
  desc.add<std::vector<std::string>>("SeedRecHitFlagToBeExcludedEE", {});
  desc.add<std::vector<std::string>>("RecHitFlagToBeExcludedEB", {});
  desc.add<std::vector<std::string>>("RecHitFlagToBeExcludedEE", {});
  descriptions.add("IslandClusterProducer", desc);
}

void IslandClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  clusterizeECALPart(
      evt, es, endcapRecHits_, endcapClusterCollection_, endcapClusterShapeAssociation_, IslandClusterAlgo::endcap);
  clusterizeECALPart(
      evt, es, barrelRecHits_, barrelClusterCollection_, barrelClusterShapeAssociation_, IslandClusterAlgo::barrel);
  nEvt_++;
}

const EcalRecHitCollection* IslandClusterProducer::getCollection(edm::Event& evt,
                                                                 const edm::EDGetTokenT<EcalRecHitCollection>& token) {
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByToken(token, rhcHandle);
  return rhcHandle.product();
}

void IslandClusterProducer::clusterizeECALPart(edm::Event& evt,
                                               const edm::EventSetup& es,
                                               const edm::EDGetTokenT<EcalRecHitCollection>& token,
                                               const std::string& clusterCollection,
                                               const std::string& clusterShapeAssociation,
                                               const IslandClusterAlgo::EcalPart& ecalPart) {
  // get the hit collection from the event:
  const EcalRecHitCollection* hitCollection_p = getCollection(evt, token);

  // get the geometry and topology from the event setup:
  edm::ESHandle<CaloGeometry> geoHandle = es.getHandle(caloGeometryToken_);

  const CaloSubdetectorGeometry* geometry_p;
  std::unique_ptr<CaloSubdetectorTopology> topology_p;

  std::string clustershapetag;
  if (ecalPart == IslandClusterAlgo::barrel) {
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
  clusters = island_p->makeClusters(hitCollection_p, geometry_p, topology_p.get(), geometryES_p, ecalPart);

  //Create associated ClusterShape objects.
  std::vector<reco::ClusterShape> ClusVec;
  for (int erg = 0; erg < int(clusters.size()); ++erg) {
    reco::ClusterShape TestShape = shapeAlgo_.Calculate(clusters[erg], hitCollection_p, geometry_p, topology_p.get());
    ClusVec.push_back(TestShape);
  }

  //Put clustershapes in event, but retain a Handle on them.
  auto clustersshapes_p = std::make_unique<reco::ClusterShapeCollection>();
  clustersshapes_p->assign(ClusVec.begin(), ClusVec.end());
  edm::OrphanHandle<reco::ClusterShapeCollection> clusHandle;
  if (ecalPart == IslandClusterAlgo::barrel)
    clusHandle = evt.put(std::move(clustersshapes_p), clustershapecollectionEB_);
  else
    clusHandle = evt.put(std::move(clustersshapes_p), clustershapecollectionEE_);

  // create a unique_ptr to a BasicClusterCollection, copy the barrel clusters into it and put in the Event:
  auto clusters_p = std::make_unique<reco::BasicClusterCollection>();
  clusters_p->assign(clusters.begin(), clusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle;
  if (ecalPart == IslandClusterAlgo::barrel)
    bccHandle = evt.put(std::move(clusters_p), barrelClusterCollection_);
  else
    bccHandle = evt.put(std::move(clusters_p), endcapClusterCollection_);

  // BasicClusterShapeAssociationMap
  auto shapeAssocs_p = std::make_unique<reco::BasicClusterShapeAssociationCollection>(bccHandle, clusHandle);
  for (unsigned int i = 0; i < clusHandle->size(); i++) {
    shapeAssocs_p->insert(edm::Ref<reco::BasicClusterCollection>(bccHandle, i),
                          edm::Ref<reco::ClusterShapeCollection>(clusHandle, i));
  }
  evt.put(std::move(shapeAssocs_p), clusterShapeAssociation);
}
