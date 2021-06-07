#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Math/interface/RectangularEtaPhiRegion.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEcal/EgammaClusterAlgos/interface/HybridClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include <iostream>
#include <memory>
#include <vector>

class HybridClusterProducer : public edm::stream::EDProducer<> {
public:
  HybridClusterProducer(const edm::ParameterSet& ps);

  ~HybridClusterProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  int nEvt_;  // internal counter of events

  std::string basicclusterCollection_;
  std::string superclusterCollection_;

  edm::EDGetTokenT<EcalRecHitCollection> hitsToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geoToken_;
  edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> sevLvToken_;

  HybridClusterAlgo* hybrid_p;  // clustering algorithm
  PositionCalc posCalculator_;  // position calculation algorithm
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HybridClusterProducer);

HybridClusterProducer::HybridClusterProducer(const edm::ParameterSet& ps) {
  basicclusterCollection_ = ps.getParameter<std::string>("basicclusterCollection");
  superclusterCollection_ = ps.getParameter<std::string>("superclusterCollection");
  hitsToken_ = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("recHitsCollection"));
  geoToken_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  sevLvToken_ = esConsumes<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd>();

  //Setup for core tools objects.
  edm::ParameterSet posCalcParameters = ps.getParameter<edm::ParameterSet>("posCalcParameters");

  posCalculator_ = PositionCalc(posCalcParameters);

  const std::vector<std::string> flagnames = ps.getParameter<std::vector<std::string> >("RecHitFlagToBeExcluded");

  const std::vector<int> flagsexcl = StringToEnumValue<EcalRecHit::Flags>(flagnames);

  const std::vector<std::string> severitynames =
      ps.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcluded");

  const std::vector<int> severitiesexcl = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynames);

  hybrid_p = new HybridClusterAlgo(ps.getParameter<double>("HybridBarrelSeedThr"),
                                   ps.getParameter<int>("step"),
                                   ps.getParameter<double>("ethresh"),
                                   ps.getParameter<double>("eseed"),
                                   ps.getParameter<double>("xi"),
                                   ps.getParameter<bool>("useEtForXi"),
                                   ps.getParameter<double>("ewing"),
                                   flagsexcl,
                                   posCalculator_,
                                   ps.getParameter<bool>("dynamicEThresh"),
                                   ps.getParameter<double>("eThreshA"),
                                   ps.getParameter<double>("eThreshB"),
                                   severitiesexcl,
                                   ps.getParameter<bool>("excludeFlagged"));
  //bremRecoveryPset,

  // get brem recovery parameters
  bool dynamicPhiRoad = ps.getParameter<bool>("dynamicPhiRoad");
  if (dynamicPhiRoad) {
    edm::ParameterSet bremRecoveryPset = ps.getParameter<edm::ParameterSet>("bremRecoveryPset");
    hybrid_p->setDynamicPhiRoad(bremRecoveryPset);
  }

  produces<reco::BasicClusterCollection>(basicclusterCollection_);
  produces<reco::SuperClusterCollection>(superclusterCollection_);
  nEvt_ = 0;
}

HybridClusterProducer::~HybridClusterProducer() { delete hybrid_p; }

void HybridClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;

  evt.getByToken(hitsToken_, rhcHandle);
  if (!(rhcHandle.isValid())) {
    edm::LogError("MissingProduct") << "could not get a handle on the EcalRecHitCollection!";
    return;
  }
  const EcalRecHitCollection* hit_collection = rhcHandle.product();

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle = es.getHandle(geoToken_);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry* geometry_p;
  std::unique_ptr<const CaloSubdetectorTopology> topology;

  edm::ESHandle<EcalSeverityLevelAlgo> sevLv = es.getHandle(sevLvToken_);

  geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  topology = std::make_unique<EcalBarrelTopology>(*geoHandle);

  // make the Basic clusters!
  reco::BasicClusterCollection basicClusters;
  hybrid_p->makeClusters(
      hit_collection, geometry_p, basicClusters, sevLv.product(), false, std::vector<RectangularEtaPhiRegion>());

  LogTrace("EcalClusters") << "Finished clustering - BasicClusterCollection returned to producer...";

  // create a unique_ptr to a BasicClusterCollection, copy the clusters into it and put in the Event:
  auto basicclusters_p = std::make_unique<reco::BasicClusterCollection>();
  basicclusters_p->assign(basicClusters.begin(), basicClusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle =
      evt.put(std::move(basicclusters_p), basicclusterCollection_);

  //Basic clusters now in the event.
  LogTrace("EcalClusters") << "Basic Clusters now put into event.";

  //Weird though it is, get the BasicClusters back out of the event.  We need the
  //edm::Ref to these guys to make our superclusters for Hybrid.

  if (!(bccHandle.isValid())) {
    edm::LogError("Missing Product") << "could not get a handle on the BasicClusterCollection!";
    return;
  }

  reco::BasicClusterCollection clusterCollection = *bccHandle;

  LogTrace("EcalClusters") << "Got the BasicClusterCollection" << std::endl;

  reco::CaloClusterPtrVector clusterPtrVector;
  for (unsigned int i = 0; i < clusterCollection.size(); i++) {
    clusterPtrVector.push_back(reco::CaloClusterPtr(bccHandle, i));
  }

  reco::SuperClusterCollection superClusters = hybrid_p->makeSuperClusters(clusterPtrVector);
  LogTrace("EcalClusters") << "Found: " << superClusters.size() << " superclusters.";

  auto superclusters_p = std::make_unique<reco::SuperClusterCollection>();
  superclusters_p->assign(superClusters.begin(), superClusters.end());

  evt.put(std::move(superclusters_p), superclusterCollection_);
  LogTrace("EcalClusters") << "Hybrid Clusters (Basic/Super) added to the Event! :-)";

  nEvt_++;
}
