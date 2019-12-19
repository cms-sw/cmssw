// -*- C++ -*-
//
// Package:    ElectronProducers
// Class:      ElectronSeedProducer
//
/**\class ElectronSeedProducer RecoEgamma/ElectronProducers/src/ElectronSeedProducer.cc

 Description: EDProducer of ElectronSeed objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronSeedGenerator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/SeedFilter.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/ClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"

class ElectronSeedProducer : public edm::stream::EDProducer<> {
public:
  explicit ElectronSeedProducer(const edm::ParameterSet&);

  void produce(edm::Event&, const edm::EventSetup&) final;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  reco::SuperClusterRefVector filterClusters(math::XYZPoint const& beamSpotPosition,
                                             const edm::Handle<reco::SuperClusterCollection>& superClusters) const;

  edm::EDGetTokenT<reco::SuperClusterCollection> superClusters_[2];
  std::vector<edm::EDGetTokenT<TrajectorySeedCollection>> initialSeeds_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;

  std::unique_ptr<ElectronSeedGenerator> matcher_;
  std::unique_ptr<SeedFilter> seedFilter_;

  bool applyHOverECut_ = true;
  std::unique_ptr<ElectronHcalHelper> hcalHelper_ = nullptr;
  double maxHOverEBarrel_;
  double maxHOverEEndcaps_;
  double maxHBarrel_;
  double maxHEndcaps_;
  double SCEtCut_;

  bool prefilteredSeeds_;

  bool allowHGCal_;
  std::unique_ptr<hgcal::ClusterTools> hgcClusterTools_;
};

using namespace reco;

ElectronSeedProducer::ElectronSeedProducer(const edm::ParameterSet& iConfig) {
  auto const& conf = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration");

  auto legacyConfSeeds = conf.getParameter<edm::InputTag>("initialSeeds");
  if (legacyConfSeeds.label().empty()) {  //new format
    initialSeeds_ =
        edm::vector_transform(conf.getParameter<std::vector<edm::InputTag>>("initialSeedsVector"),
                              [this](edm::InputTag const& tag) { return consumes<TrajectorySeedCollection>(tag); });
  } else {
    initialSeeds_ = {consumes<TrajectorySeedCollection>(conf.getParameter<edm::InputTag>("initialSeeds"))};
  }

  SCEtCut_ = conf.getParameter<double>("SCEtCut");
  prefilteredSeeds_ = conf.getParameter<bool>("preFilteredSeeds");

  auto theconsumes = consumesCollector();

  // new beamSpot tag
  beamSpotTag_ = consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpot"));

  // for H/E
  applyHOverECut_ = conf.getParameter<bool>("applyHOverECut");
  if (applyHOverECut_) {
    ElectronHcalHelper::Configuration hcalCfg;
    hcalCfg.hOverEConeSize = conf.getParameter<double>("hOverEConeSize");
    if (hcalCfg.hOverEConeSize > 0) {
      hcalCfg.useTowers = true;
      hcalCfg.hcalTowers = consumes<CaloTowerCollection>(conf.getParameter<edm::InputTag>("hcalTowers"));
      hcalCfg.hOverEPtMin = conf.getParameter<double>("hOverEPtMin");
    }
    hcalHelper_ = std::make_unique<ElectronHcalHelper>(hcalCfg);

    allowHGCal_ = conf.getParameter<bool>("allowHGCal");
    if (allowHGCal_) {
      const edm::ParameterSet& hgcCfg = conf.getParameterSet("HGCalConfig");
      hgcClusterTools_ = std::make_unique<hgcal::ClusterTools>(hgcCfg, theconsumes);
    }

    maxHOverEBarrel_ = conf.getParameter<double>("maxHOverEBarrel");
    maxHOverEEndcaps_ = conf.getParameter<double>("maxHOverEEndcaps");
    maxHBarrel_ = conf.getParameter<double>("maxHBarrel");
    maxHEndcaps_ = conf.getParameter<double>("maxHEndcaps");
  }

  edm::ParameterSet rpset = conf.getParameter<edm::ParameterSet>("RegionPSet");

  ElectronSeedGenerator::Tokens esg_tokens;
  esg_tokens.token_bs = beamSpotTag_;
  esg_tokens.token_vtx = mayConsume<reco::VertexCollection>(conf.getParameter<edm::InputTag>("vertices"));

  matcher_ = std::make_unique<ElectronSeedGenerator>(conf, esg_tokens);

  superClusters_[0] =
      consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("barrelSuperClusters"));
  superClusters_[1] =
      consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("endcapSuperClusters"));

  // Construction of SeedFilter was in beginRun() with the comment
  // below, but it has to be done here because of ConsumesCollector
  //
  // FIXME: because of a bug presumably in tracker seeding,
  // perhaps in CombinedHitPairGenerator, badly caching some EventSetup product,
  // we must redo the SeedFilter for each run.
  if (prefilteredSeeds_) {
    SeedFilter::Tokens sf_tokens;
    sf_tokens.token_bs = beamSpotTag_;
    sf_tokens.token_vtx = consumes<std::vector<reco::Vertex>>(rpset.getParameter<edm::InputTag>("VertexProducer"));

    edm::ConsumesCollector iC = consumesCollector();
    seedFilter_ = std::make_unique<SeedFilter>(conf, sf_tokens, iC);
  }

  //register your products
  produces<ElectronSeedCollection>();
}

void ElectronSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) {
  LogDebug("ElectronSeedProducer") << "[ElectronSeedProducer::produce] entering ";

  std::vector<TrajectorySeedCollection const*> initialSeedCollections;
  std::unique_ptr<TrajectorySeedCollection> initialSeedCollectionPtr = nullptr;  //created on the fly

  if (hcalHelper_) {
    hcalHelper_->checkSetup(iSetup);
    hcalHelper_->readEvent(e);
    if (allowHGCal_) {
      hgcClusterTools_->getEventSetup(iSetup);
      hgcClusterTools_->getEvent(e);
    }
  }

  matcher_->setupES(iSetup);

  // get initial TrajectorySeeds if necessary
  if (!prefilteredSeeds_) {
    initialSeedCollections.clear();
    for (auto const& seeds : initialSeeds_) {
      initialSeedCollections.push_back(&e.get(seeds));
    }
  } else {
    initialSeedCollections.clear();  //reset later
    initialSeedCollectionPtr = std::make_unique<TrajectorySeedCollection>();
  }

  auto seeds = std::make_unique<ElectronSeedCollection>();
  auto const& beamSportPosition = e.get(beamSpotTag_).position();

  // loop over barrel + endcap
  for (unsigned int i = 0; i < 2; i++) {
    auto clusterRefs = filterClusters(beamSportPosition, e.getHandle(superClusters_[i]));
    if (prefilteredSeeds_) {
      for (auto const& sclRef : clusterRefs) {
        seedFilter_->seeds(e, iSetup, sclRef, initialSeedCollectionPtr.get());
        initialSeedCollections.push_back(initialSeedCollectionPtr.get());
        LogDebug("ElectronSeedProducer") << "Number of Seeds: " << initialSeedCollections.back()->size();
      }
    }
    matcher_->run(e, iSetup, clusterRefs, initialSeedCollections, *seeds);
  }

  // store the accumulated result
#ifdef EDM_ML_DEBUG
  for (auto const& seed : *seeds) {
    SuperClusterRef superCluster = seed.caloCluster().castTo<SuperClusterRef>();
    LogDebug("ElectronSeedProducer") << "new seed with " << seed.nHits() << " hits"
                                     << ", charge " << seed.getCharge() << " and cluster energy "
                                     << superCluster->energy() << " PID " << superCluster.id();
  }
#endif
  e.put(std::move(seeds));
}

//===============================
// Filter the superclusters
// - with EtCut
// - with HoE using calo cone
//===============================

SuperClusterRefVector ElectronSeedProducer::filterClusters(
    math::XYZPoint const& beamSpotPosition, const edm::Handle<reco::SuperClusterCollection>& superClusters) const {
  SuperClusterRefVector sclRefs;

  for (unsigned int i = 0; i < superClusters->size(); ++i) {
    auto const& scl = (*superClusters)[i];
    double sclEta = EleRelPoint(scl.position(), beamSpotPosition).eta();
    if (scl.energy() / cosh(sclEta) > SCEtCut_) {
      if (applyHOverECut_) {
        bool hoeVeto = false;
        double had = hcalHelper_->hcalESumDepth1(scl) + hcalHelper_->hcalESumDepth2(scl);
        double scle = scl.energy();
        int det_group = scl.seed()->hitsAndFractions()[0].first.det();
        int detector = scl.seed()->hitsAndFractions()[0].first.subdetId();
        if (detector == EcalBarrel && (had < maxHBarrel_ || had / scle < maxHOverEBarrel_))
          hoeVeto = true;
        else if (!allowHGCal_ && detector == EcalEndcap && (had < maxHEndcaps_ || had / scle < maxHOverEEndcaps_))
          hoeVeto = true;
        else if (allowHGCal_ && EcalTools::isHGCalDet((DetId::Detector)det_group)) {
          float had_fraction = hgcClusterTools_->getClusterHadronFraction(*(scl.seed()));
          hoeVeto = (had_fraction >= 0.f && had_fraction < maxHOverEEndcaps_);
        }
        if (hoeVeto) {
          sclRefs.push_back({superClusters, i});
        }
      } else {
        sclRefs.push_back({superClusters, i});
      }
    }
  }
  LogDebug("ElectronSeedProducer") << "Filtered out " << sclRefs.size() << " superclusters from "
                                   << superClusters->size();

  return sclRefs;
}

void ElectronSeedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("endcapSuperClusters",
                          {"particleFlowSuperClusterECAL", "particleFlowSuperClusterECALEndcapWithPreshower"});
  {
    edm::ParameterSetDescription psd0, psd1, psd2, psd3;
    psd1.add<std::string>("ComponentName", "StandardHitPairGenerator");
    psd1.addUntracked<int>("useOnDemandTracker", 0);
    psd0.add<edm::ParameterSetDescription>("OrderedHitsFactoryPSet", psd1);

    psd2.add<double>("deltaPhiRegion", 0.4);
    psd2.add<double>("originHalfLength", 15.0);
    psd2.add<bool>("useZInVertex", true);
    psd2.add<double>("deltaEtaRegion", 0.1);
    psd2.add<double>("ptMin", 1.5);
    psd2.add<double>("originRadius", 0.2);
    psd2.add<edm::InputTag>("VertexProducer", {"dummyVertices"});
    psd0.add<edm::ParameterSetDescription>("RegionPSet", psd2);

    // steering
    psd0.add<edm::InputTag>("initialSeeds", {""});  //keep for be compatibility
    psd0.add<std::vector<edm::InputTag>>("initialSeedsVector", {});
    psd0.add<bool>("preFilteredSeeds", false);
    psd0.add<bool>("useRecoVertex", false);
    psd0.add<edm::InputTag>("vertices", {"offlinePrimaryVerticesWithBS"});
    psd0.add<edm::InputTag>("beamSpot", {"offlineBeamSpot"});
    psd0.add<bool>("dynamicPhiRoad", true);

    // specify where to get the hits from
    psd0.add<edm::InputTag>("measurementTrackerEvent", {"MeasurementTrackerEvent"});

    // SC filtering
    psd0.add<double>("SCEtCut", 0.0);

    // H/E
    psd0.add<bool>("applyHOverECut", true);
    psd0.add<double>("hOverEConeSize", 0.15);
    psd0.add<double>("maxHOverEBarrel", 0.15);
    psd0.add<double>("maxHOverEEndcaps", 0.15);
    psd0.add<double>("maxHBarrel", 0.0);
    psd0.add<double>("maxHEndcaps", 0.0);

    // H/E towers
    psd0.add<edm::InputTag>("hcalTowers", {"towerMaker"});
    psd0.add<double>("hOverEPtMin", 0.0);

    // H/E equivalent for HGCal
    psd0.add<bool>("allowHGCal", false);
    edm::ParameterSetDescription psd4;
    psd4.add<edm::InputTag>("HGCEEInput", {"HGCalRecHit", "HGCEERecHits"});
    psd4.add<edm::InputTag>("HGCFHInput", {"HGCalRecHit", "HGCHEFRecHits"});
    psd4.add<edm::InputTag>("HGCBHInput", {"HGCalRecHit", "HGCHEBRecHits"});
    psd0.add<edm::ParameterSetDescription>("HGCalConfig", psd4);

    // r/z windows
    psd0.add<double>("nSigmasDeltaZ1", 5.0);      // in case beam spot is used for the matching
    psd0.add<double>("deltaZ1WithVertex", 25.0);  // in case reco vertex is used for the matching
    psd0.add<double>("z2MinB", -0.09);
    psd0.add<double>("z2MaxB", 0.09);
    psd0.add<double>("r2MinF", -0.15);
    psd0.add<double>("r2MaxF", 0.15);
    psd0.add<double>("rMinI", -0.2);  // intermediate region SC in EB and 2nd hits in PXF
    psd0.add<double>("rMaxI", 0.2);   // intermediate region SC in EB and 2nd hits in PXF

    // phi windows (dynamic)
    psd0.add<double>("LowPtThreshold", 5.0);
    psd0.add<double>("HighPtThreshold", 35.0);
    psd0.add<double>("SizeWindowENeg", 0.675);
    psd0.add<double>("DeltaPhi1Low", 0.23);
    psd0.add<double>("DeltaPhi1High", 0.08);
    psd0.add<double>("DeltaPhi2B", 0.008);
    psd0.add<double>("DeltaPhi2F", 0.012);

    // phi windows (non dynamic, overwritten in case dynamic is selected)
    psd0.add<double>("ePhiMin1", -0.125);
    psd0.add<double>("ePhiMax1", 0.075);
    psd0.add<double>("pPhiMin1", -0.075);
    psd0.add<double>("pPhiMax1", 0.125);
    psd0.add<double>("PhiMin2B", -0.002);
    psd0.add<double>("PhiMax2B", 0.002);
    psd0.add<double>("PhiMin2F", -0.003);
    psd0.add<double>("PhiMax2F", 0.003);

    psd3.add<std::string>("ComponentName", "SeedFromConsecutiveHitsCreator");
    psd3.add<std::string>("propagator", "PropagatorWithMaterial");
    psd3.add<double>("SeedMomentumForBOFF", 5.0);
    psd3.add<double>("OriginTransverseErrorMultiplier", 1.0);
    psd3.add<double>("MinOneOverPtError", 1.0);
    psd3.add<std::string>("magneticField", "");
    psd3.add<std::string>("TTRHBuilder", "WithTrackAngle");
    psd3.add<bool>("forceKinematicWithRegionDirection", false);

    psd0.add<edm::ParameterSetDescription>("SeedCreatorPSet", psd3);

    desc.add<edm::ParameterSetDescription>("SeedConfiguration", psd0);
  }
  desc.add<edm::InputTag>("barrelSuperClusters",
                          {"particleFlowSuperClusterECAL", "particleFlowSuperClusterECALBarrel"});
  descriptions.add("ecalDrivenElectronSeeds", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronSeedProducer);
