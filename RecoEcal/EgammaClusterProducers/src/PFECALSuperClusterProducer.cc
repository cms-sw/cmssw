#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "RecoEcal/EgammaClusterAlgos/interface/PFECALSuperClusterAlgo.h"
#include "RecoEcal/EgammaClusterAlgos/interface/SCEnergyCorrectorSemiParm.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/SCProducerCache.h"
#include "TVector2.h"

#include <memory>
#include <vector>

/*
 * class PFECALSuperClusterProducer 
 * author Nicolas Chanon
 * Additional authors for Mustache: Y. Gershtein, R. Patel, L. Gray
 * Additional authors for DeepSC: D.Valsecchi, B.Marzocchi
 * date   July 2012
 * updates Feb 2022
 */

class PFECALSuperClusterProducer : public edm::stream::EDProducer<edm::GlobalCache<reco::SCProducerCache>> {
public:
  explicit PFECALSuperClusterProducer(const edm::ParameterSet&, const reco::SCProducerCache* gcache);
  ~PFECALSuperClusterProducer() override;

  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  static std::unique_ptr<reco::SCProducerCache> initializeGlobalCache(const edm::ParameterSet& config) {
    return std::make_unique<reco::SCProducerCache>(config);
  }

  static void globalEndJob(const reco::SCProducerCache*){};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------member data ---------------------------

  /// clustering algorithm
  PFECALSuperClusterAlgo superClusterAlgo_;
  PFECALSuperClusterAlgo::clustering_type _theclusteringtype;
  PFECALSuperClusterAlgo::energy_weight _theenergyweight;

  std::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_;

  /// verbose ?
  bool verbose_;

  std::string PFBasicClusterCollectionBarrel_;
  std::string PFSuperClusterCollectionBarrel_;
  std::string PFBasicClusterCollectionEndcap_;
  std::string PFSuperClusterCollectionEndcap_;
  std::string PFBasicClusterCollectionPreshower_;
  std::string PFSuperClusterCollectionEndcapWithPreshower_;
  std::string PFClusterAssociationEBEE_;
  std::string PFClusterAssociationES_;

  // OOT photons
  bool isOOTCollection_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFECALSuperClusterProducer);

using namespace std;

using namespace edm;

namespace {
  const std::string ClusterType__BOX("Box");
  const std::string ClusterType__Mustache("Mustache");
  const std::string ClusterType__DeepSC("DeepSC");

  const std::string EnergyWeight__Raw("Raw");
  const std::string EnergyWeight__CalibratedNoPS("CalibratedNoPS");
  const std::string EnergyWeight__CalibratedTotal("CalibratedTotal");
}  // namespace

PFECALSuperClusterProducer::PFECALSuperClusterProducer(const edm::ParameterSet& iConfig,
                                                       const reco::SCProducerCache* gcache)
    : superClusterAlgo_(gcache) {
  verbose_ = iConfig.getUntrackedParameter<bool>("verbose", false);

  superClusterAlgo_.setUseRegression(iConfig.getParameter<bool>("useRegression"));

  isOOTCollection_ = iConfig.getParameter<bool>("isOOTCollection");
  superClusterAlgo_.setIsOOTCollection(isOOTCollection_);

  std::string _typename = iConfig.getParameter<std::string>("ClusteringType");
  if (_typename == ClusterType__BOX) {
    _theclusteringtype = PFECALSuperClusterAlgo::kBOX;
  } else if (_typename == ClusterType__Mustache) {
    _theclusteringtype = PFECALSuperClusterAlgo::kMustache;
  } else if (_typename == ClusterType__DeepSC) {
    _theclusteringtype = PFECALSuperClusterAlgo::kDeepSC;
  } else {
    throw cms::Exception("InvalidClusteringType") << "You have not chosen a valid clustering type,"
                                                  << " please choose from \"Box\" or \"Mustache\" or \"DeepSC\"!";
  }

  std::string _weightname = iConfig.getParameter<std::string>("EnergyWeight");
  if (_weightname == EnergyWeight__Raw) {
    _theenergyweight = PFECALSuperClusterAlgo::kRaw;
  } else if (_weightname == EnergyWeight__CalibratedNoPS) {
    _theenergyweight = PFECALSuperClusterAlgo::kCalibratedNoPS;
  } else if (_weightname == EnergyWeight__CalibratedTotal) {
    _theenergyweight = PFECALSuperClusterAlgo::kCalibratedTotal;
  } else {
    throw cms::Exception("InvalidClusteringType") << "You have not chosen a valid energy weighting scheme,"
                                                  << " please choose from \"Raw\", \"CalibratedNoPS\", or"
                                                  << " \"CalibratedTotal\"!";
  }

  // parameters for clustering
  bool seedThresholdIsET = iConfig.getParameter<bool>("seedThresholdIsET");

  bool useDynamicDPhi = iConfig.getParameter<bool>("useDynamicDPhiWindow");

  double threshPFClusterSeedBarrel = iConfig.getParameter<double>("thresh_PFClusterSeedBarrel");
  double threshPFClusterBarrel = iConfig.getParameter<double>("thresh_PFClusterBarrel");

  double threshPFClusterSeedEndcap = iConfig.getParameter<double>("thresh_PFClusterSeedEndcap");
  double threshPFClusterEndcap = iConfig.getParameter<double>("thresh_PFClusterEndcap");

  double phiwidthSuperClusterBarrel = iConfig.getParameter<double>("phiwidth_SuperClusterBarrel");
  double etawidthSuperClusterBarrel = iConfig.getParameter<double>("etawidth_SuperClusterBarrel");

  double phiwidthSuperClusterEndcap = iConfig.getParameter<double>("phiwidth_SuperClusterEndcap");
  double etawidthSuperClusterEndcap = iConfig.getParameter<double>("etawidth_SuperClusterEndcap");

  double doSatelliteClusterMerge = iConfig.getParameter<bool>("doSatelliteClusterMerge");
  double satelliteClusterSeedThreshold = iConfig.getParameter<double>("satelliteClusterSeedThreshold");
  double satelliteMajorityFraction = iConfig.getParameter<double>("satelliteMajorityFraction");
  bool dropUnseedable = iConfig.getParameter<bool>("dropUnseedable");

  superClusterAlgo_.setClusteringType(_theclusteringtype);
  superClusterAlgo_.setUseDynamicDPhi(useDynamicDPhi);
  // clusteringType and useDynamicDPhi need to be defined before setting the tokens in order to esConsume only the necessary records
  superClusterAlgo_.setTokens(iConfig, consumesCollector());

  superClusterAlgo_.setVerbosityLevel(verbose_);
  superClusterAlgo_.setEnergyWeighting(_theenergyweight);
  superClusterAlgo_.setUseETForSeeding(seedThresholdIsET);

  superClusterAlgo_.setThreshSuperClusterEt(iConfig.getParameter<double>("thresh_SCEt"));

  superClusterAlgo_.setThreshPFClusterSeedBarrel(threshPFClusterSeedBarrel);
  superClusterAlgo_.setThreshPFClusterBarrel(threshPFClusterBarrel);

  superClusterAlgo_.setThreshPFClusterSeedEndcap(threshPFClusterSeedEndcap);
  superClusterAlgo_.setThreshPFClusterEndcap(threshPFClusterEndcap);

  superClusterAlgo_.setPhiwidthSuperClusterBarrel(phiwidthSuperClusterBarrel);
  superClusterAlgo_.setEtawidthSuperClusterBarrel(etawidthSuperClusterBarrel);

  superClusterAlgo_.setPhiwidthSuperClusterEndcap(phiwidthSuperClusterEndcap);
  superClusterAlgo_.setEtawidthSuperClusterEndcap(etawidthSuperClusterEndcap);

  superClusterAlgo_.setSatelliteMerging(doSatelliteClusterMerge);
  superClusterAlgo_.setSatelliteThreshold(satelliteClusterSeedThreshold);
  superClusterAlgo_.setMajorityFraction(satelliteMajorityFraction);
  superClusterAlgo_.setDropUnseedable(dropUnseedable);

  //Load the ECAL energy calibration
  thePFEnergyCalibration_ = std::make_shared<PFEnergyCalibration>();
  superClusterAlgo_.setPFClusterCalibration(thePFEnergyCalibration_);

  bool applyCrackCorrections_ = iConfig.getParameter<bool>("applyCrackCorrections");
  superClusterAlgo_.setCrackCorrections(applyCrackCorrections_);

  PFBasicClusterCollectionBarrel_ = iConfig.getParameter<string>("PFBasicClusterCollectionBarrel");
  PFSuperClusterCollectionBarrel_ = iConfig.getParameter<string>("PFSuperClusterCollectionBarrel");

  PFBasicClusterCollectionEndcap_ = iConfig.getParameter<string>("PFBasicClusterCollectionEndcap");
  PFSuperClusterCollectionEndcap_ = iConfig.getParameter<string>("PFSuperClusterCollectionEndcap");

  PFBasicClusterCollectionPreshower_ = iConfig.getParameter<string>("PFBasicClusterCollectionPreshower");
  PFSuperClusterCollectionEndcapWithPreshower_ =
      iConfig.getParameter<string>("PFSuperClusterCollectionEndcapWithPreshower");

  PFClusterAssociationEBEE_ = "PFClusterAssociationEBEE";
  PFClusterAssociationES_ = "PFClusterAssociationES";

  produces<reco::SuperClusterCollection>(PFSuperClusterCollectionBarrel_);
  produces<reco::SuperClusterCollection>(PFSuperClusterCollectionEndcapWithPreshower_);
  produces<reco::CaloClusterCollection>(PFBasicClusterCollectionBarrel_);
  produces<reco::CaloClusterCollection>(PFBasicClusterCollectionEndcap_);
  produces<reco::CaloClusterCollection>(PFBasicClusterCollectionPreshower_);
  produces<edm::ValueMap<reco::CaloClusterPtr>>(PFClusterAssociationEBEE_);
  produces<edm::ValueMap<reco::CaloClusterPtr>>(PFClusterAssociationES_);
}

PFECALSuperClusterProducer::~PFECALSuperClusterProducer() {}

void PFECALSuperClusterProducer::beginLuminosityBlock(LuminosityBlock const& iL, EventSetup const& iE) {
  superClusterAlgo_.update(iE);
}

void PFECALSuperClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // update SC parameters
  superClusterAlgo_.updateSCParams(iSetup);
  // do clustering
  superClusterAlgo_.loadAndSortPFClusters(iEvent);
  superClusterAlgo_.run(iEvent);

  //build collections of output CaloClusters from the used PFClusters
  auto caloClustersEB = std::make_unique<reco::CaloClusterCollection>();
  auto caloClustersEE = std::make_unique<reco::CaloClusterCollection>();
  auto caloClustersES = std::make_unique<reco::CaloClusterCollection>();

  std::map<reco::CaloClusterPtr, unsigned int> pfClusterMapEB;  //maps of pfclusters to caloclusters
  std::map<reco::CaloClusterPtr, unsigned int> pfClusterMapEE;
  std::map<reco::CaloClusterPtr, unsigned int> pfClusterMapES;

  //fill calocluster collections and maps
  for (const auto& ebsc : *(superClusterAlgo_.getEBOutputSCCollection())) {
    for (reco::CaloCluster_iterator pfclus = ebsc.clustersBegin(); pfclus != ebsc.clustersEnd(); ++pfclus) {
      if (!pfClusterMapEB.count(*pfclus)) {
        reco::CaloCluster caloclus(**pfclus);
        caloClustersEB->push_back(caloclus);
        pfClusterMapEB[*pfclus] = caloClustersEB->size() - 1;
      } else {
        throw cms::Exception("PFECALSuperClusterProducer::produce")
            << "Found an EB pfcluster matched to more than one EB supercluster!" << std::dec << std::endl;
      }
    }
  }
  for (const auto& eesc : *(superClusterAlgo_.getEEOutputSCCollection())) {
    for (reco::CaloCluster_iterator pfclus = eesc.clustersBegin(); pfclus != eesc.clustersEnd(); ++pfclus) {
      if (!pfClusterMapEE.count(*pfclus)) {
        reco::CaloCluster caloclus(**pfclus);
        caloClustersEE->push_back(caloclus);
        pfClusterMapEE[*pfclus] = caloClustersEE->size() - 1;
      } else {
        throw cms::Exception("PFECALSuperClusterProducer::produce")
            << "Found an EE pfcluster matched to more than one EE supercluster!" << std::dec << std::endl;
      }
    }
    for (reco::CaloCluster_iterator pfclus = eesc.preshowerClustersBegin(); pfclus != eesc.preshowerClustersEnd();
         ++pfclus) {
      if (!pfClusterMapES.count(*pfclus)) {
        reco::CaloCluster caloclus(**pfclus);
        caloClustersES->push_back(caloclus);
        pfClusterMapES[*pfclus] = caloClustersES->size() - 1;
      } else {
        throw cms::Exception("PFECALSuperClusterProducer::produce")
            << "Found an ES pfcluster matched to more than one EE supercluster!" << std::dec << std::endl;
      }
    }
  }

  //create ValueMaps from output CaloClusters back to original PFClusters
  auto pfClusterAssociationEBEE = std::make_unique<edm::ValueMap<reco::CaloClusterPtr>>();
  auto pfClusterAssociationES = std::make_unique<edm::ValueMap<reco::CaloClusterPtr>>();

  //vectors to fill ValueMaps
  std::vector<reco::CaloClusterPtr> clusptrsEB(caloClustersEB->size());
  std::vector<reco::CaloClusterPtr> clusptrsEE(caloClustersEE->size());
  std::vector<reco::CaloClusterPtr> clusptrsES(caloClustersES->size());

  //put calocluster output collections in event and get orphan handles to create ptrs
  const edm::OrphanHandle<reco::CaloClusterCollection>& caloClusHandleEB =
      iEvent.put(std::move(caloClustersEB), PFBasicClusterCollectionBarrel_);
  const edm::OrphanHandle<reco::CaloClusterCollection>& caloClusHandleEE =
      iEvent.put(std::move(caloClustersEE), PFBasicClusterCollectionEndcap_);
  const edm::OrphanHandle<reco::CaloClusterCollection>& caloClusHandleES =
      iEvent.put(std::move(caloClustersES), PFBasicClusterCollectionPreshower_);

  //relink superclusters to output caloclusters and fill vectors for ValueMaps
  for (auto& ebsc : *(superClusterAlgo_.getEBOutputSCCollection())) {
    reco::CaloClusterPtr seedptr(caloClusHandleEB, pfClusterMapEB[ebsc.seed()]);
    ebsc.setSeed(seedptr);

    reco::CaloClusterPtrVector clusters;
    for (reco::CaloCluster_iterator pfclus = ebsc.clustersBegin(); pfclus != ebsc.clustersEnd(); ++pfclus) {
      int caloclusidx = pfClusterMapEB[*pfclus];
      reco::CaloClusterPtr clusptr(caloClusHandleEB, caloclusidx);
      clusters.push_back(clusptr);
      clusptrsEB[caloclusidx] = *pfclus;
    }
    ebsc.setClusters(clusters);
  }
  for (auto& eesc : *(superClusterAlgo_.getEEOutputSCCollection())) {
    reco::CaloClusterPtr seedptr(caloClusHandleEE, pfClusterMapEE[eesc.seed()]);
    eesc.setSeed(seedptr);

    reco::CaloClusterPtrVector clusters;
    for (reco::CaloCluster_iterator pfclus = eesc.clustersBegin(); pfclus != eesc.clustersEnd(); ++pfclus) {
      int caloclusidx = pfClusterMapEE[*pfclus];
      reco::CaloClusterPtr clusptr(caloClusHandleEE, caloclusidx);
      clusters.push_back(clusptr);
      clusptrsEE[caloclusidx] = *pfclus;
    }
    eesc.setClusters(clusters);

    reco::CaloClusterPtrVector psclusters;
    for (reco::CaloCluster_iterator pfclus = eesc.preshowerClustersBegin(); pfclus != eesc.preshowerClustersEnd();
         ++pfclus) {
      int caloclusidx = pfClusterMapES[*pfclus];
      reco::CaloClusterPtr clusptr(caloClusHandleES, caloclusidx);
      psclusters.push_back(clusptr);
      clusptrsES[caloclusidx] = *pfclus;
    }
    eesc.setPreshowerClusters(psclusters);
  }

  //fill association maps from output CaloClusters back to original PFClusters
  edm::ValueMap<reco::CaloClusterPtr>::Filler fillerEBEE(*pfClusterAssociationEBEE);
  fillerEBEE.insert(caloClusHandleEB, clusptrsEB.begin(), clusptrsEB.end());
  fillerEBEE.insert(caloClusHandleEE, clusptrsEE.begin(), clusptrsEE.end());
  fillerEBEE.fill();

  edm::ValueMap<reco::CaloClusterPtr>::Filler fillerES(*pfClusterAssociationES);
  fillerES.insert(caloClusHandleES, clusptrsES.begin(), clusptrsES.end());
  fillerES.fill();

  //store in the event
  iEvent.put(std::move(pfClusterAssociationEBEE), PFClusterAssociationEBEE_);
  iEvent.put(std::move(pfClusterAssociationES), PFClusterAssociationES_);
  iEvent.put(std::move(superClusterAlgo_.getEBOutputSCCollection()), PFSuperClusterCollectionBarrel_);
  iEvent.put(std::move(superClusterAlgo_.getEEOutputSCCollection()), PFSuperClusterCollectionEndcapWithPreshower_);
}

void PFECALSuperClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("PFSuperClusterCollectionEndcap", "particleFlowSuperClusterECALEndcap");
  desc.add<bool>("doSatelliteClusterMerge", false);
  desc.add<double>("thresh_PFClusterBarrel", 0.0);
  desc.add<std::string>("PFBasicClusterCollectionBarrel", "particleFlowBasicClusterECALBarrel");
  desc.add<bool>("useRegression", true);
  // Track isolation parameters: begin
  //  desc.add<edm::InputTag>("trackProducer", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("trackProducer", edm::InputTag(""));
  desc.add<double>("trkIsoPtMin", 0.5);
  desc.add<double>("trkIsoConeSize", 0.4);
  desc.add<double>("trkIsoZSpan", 999999.9);
  desc.add<double>("trkIsoRSpan", 999999.9);
  desc.add<double>("trkIsoVetoConeSize", 0.06);
  desc.add<double>("trkIsoStripBarrel", 0.03);
  desc.add<double>("trkIsoStripEndcap", 0.03);

  // Track isolation parameters: end
  desc.add<double>("satelliteMajorityFraction", 0.5);
  desc.add<double>("thresh_PFClusterEndcap", 0.0);
  desc.add<edm::InputTag>("ESAssociation", edm::InputTag("particleFlowClusterECAL"));
  desc.add<std::string>("PFBasicClusterCollectionPreshower", "particleFlowBasicClusterECALPreshower");
  desc.addUntracked<bool>("verbose", false);
  desc.add<double>("thresh_SCEt", 4.0);
  desc.add<double>("etawidth_SuperClusterEndcap", 0.04);
  desc.add<double>("phiwidth_SuperClusterEndcap", 0.6);
  desc.add<bool>("useDynamicDPhiWindow", true);
  desc.add<std::string>("PFSuperClusterCollectionBarrel", "particleFlowSuperClusterECALBarrel");
  desc.add<edm::ParameterSetDescription>("regressionConfig", SCEnergyCorrectorSemiParm::makePSetDescription());
  desc.add<bool>("applyCrackCorrections", false);
  desc.add<double>("satelliteClusterSeedThreshold", 50.0);
  desc.add<double>("etawidth_SuperClusterBarrel", 0.04);
  desc.add<std::string>("PFBasicClusterCollectionEndcap", "particleFlowBasicClusterECALEndcap");
  desc.add<edm::InputTag>("PFClusters", edm::InputTag("particleFlowClusterECAL"));
  desc.add<double>("thresh_PFClusterSeedBarrel", 1.0);
  desc.add<std::string>("EnergyWeight", "Raw");
  desc.add<edm::InputTag>("BeamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<double>("thresh_PFClusterSeedEndcap", 1.0);
  desc.add<double>("phiwidth_SuperClusterBarrel", 0.6);
  desc.add<double>("thresh_PFClusterES", 0.0);
  desc.add<bool>("seedThresholdIsET", true);
  desc.add<bool>("isOOTCollection", false);
  desc.add<edm::InputTag>("barrelRecHits", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("endcapRecHits", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<std::string>("PFSuperClusterCollectionEndcapWithPreshower",
                        "particleFlowSuperClusterECALEndcapWithPreshower");
  desc.add<bool>("dropUnseedable", false);

  edm::ParameterSetDescription deepSCParams;
  deepSCParams.add<std::string>("modelFile", "");
  deepSCParams.add<std::string>("configFileClusterFeatures", "");
  deepSCParams.add<std::string>("configFileWindowFeatures", "");
  deepSCParams.add<std::string>("configFileHitsFeatures", "");
  deepSCParams.add<uint>("nClusterFeatures", 12);
  deepSCParams.add<uint>("nWindowFeatures", 18);
  deepSCParams.add<uint>("nHitsFeatures", 4);
  deepSCParams.add<uint>("maxNClusters", 40);
  deepSCParams.add<uint>("maxNRechits", 40);
  deepSCParams.add<uint>("batchSize", 64);
  deepSCParams.add<std::string>("collectionStrategy", "Cascade");

  EmptyGroupDescription emptyGroup;

  // Add DeepSC parameters only to the specific ClusteringType
  edm::ParameterSwitch<std::string> switchNode(
      edm::ParameterDescription<std::string>("ClusteringType", ClusterType__Mustache, true),
      ClusterType__Mustache >> emptyGroup or ClusterType__BOX >> emptyGroup or
          ClusterType__DeepSC >>
              edm::ParameterDescription<edm::ParameterSetDescription>("deepSuperClusterConfig", deepSCParams, true));
  desc.addNode(switchNode);

  descriptions.add("particleFlowSuperClusterECALMustache", desc);
}
