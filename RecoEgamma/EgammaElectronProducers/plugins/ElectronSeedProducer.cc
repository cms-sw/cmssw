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

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronSeedGenerator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
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

  bool applyHOverECut_ = true;
  std::unique_ptr<ElectronHcalHelper> hcalHelper_ = nullptr;
  double maxHOverEBarrel_;
  double maxHOverEEndcaps_;
  double SCEtCut_;

  bool allowHGCal_;
  std::unique_ptr<hgcal::ClusterTools> hgcClusterTools_;
};

using namespace reco;

ElectronSeedProducer::ElectronSeedProducer(const edm::ParameterSet& conf)
    :

      initialSeeds_{
          edm::vector_transform(conf.getParameter<std::vector<edm::InputTag>>("initialSeedsVector"),
                                [this](edm::InputTag const& tag) { return consumes<TrajectorySeedCollection>(tag); })}

{
  SCEtCut_ = conf.getParameter<double>("SCEtCut");
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
  }

  ElectronSeedGenerator::Tokens esg_tokens;
  esg_tokens.token_bs = beamSpotTag_;
  esg_tokens.token_vtx = mayConsume<reco::VertexCollection>(conf.getParameter<edm::InputTag>("vertices"));

  matcher_ = std::make_unique<ElectronSeedGenerator>(conf, esg_tokens);

  superClusters_[0] = consumes<reco::SuperClusterCollection>(conf.getParameter<edm::InputTag>("barrelSuperClusters"));
  superClusters_[1] = consumes<reco::SuperClusterCollection>(conf.getParameter<edm::InputTag>("endcapSuperClusters"));

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

  // get initial TrajectorySeeds
  initialSeedCollections.clear();
  for (auto const& seeds : initialSeeds_) {
    initialSeedCollections.push_back(&e.get(seeds));
  }

  auto seeds = std::make_unique<ElectronSeedCollection>();
  auto const& beamSportPosition = e.get(beamSpotTag_).position();

  // loop over barrel + endcap
  for (auto superCluster : superClusters_) {
    auto clusterRefs = filterClusters(beamSportPosition, e.getHandle(superCluster));
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
        if (detector == EcalBarrel && had / scle < maxHOverEBarrel_)
          hoeVeto = true;
        else if (!allowHGCal_ && detector == EcalEndcap && had / scle < maxHOverEEndcaps_)
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

  // steering
  desc.add<std::vector<edm::InputTag>>("initialSeedsVector", {});
  desc.add<bool>("useRecoVertex", false);
  desc.add<edm::InputTag>("vertices", {"offlinePrimaryVerticesWithBS"});
  desc.add<edm::InputTag>("beamSpot", {"offlineBeamSpot"});
  desc.add<bool>("dynamicPhiRoad", true);

  // SC filtering
  desc.add<double>("SCEtCut", 0.0);

  // H/E
  desc.add<bool>("applyHOverECut", true);
  desc.add<double>("hOverEConeSize", 0.15);
  desc.add<double>("maxHOverEBarrel", 0.15);
  desc.add<double>("maxHOverEEndcaps", 0.15);

  // H/E towers
  desc.add<edm::InputTag>("hcalTowers", {"towerMaker"});
  desc.add<double>("hOverEPtMin", 0.0);

  // H/E equivalent for HGCal
  desc.add<bool>("allowHGCal", false);
  edm::ParameterSetDescription psd4;
  psd4.add<edm::InputTag>("HGCEEInput", {"HGCalRecHit", "HGCEERecHits"});
  psd4.add<edm::InputTag>("HGCFHInput", {"HGCalRecHit", "HGCHEFRecHits"});
  psd4.add<edm::InputTag>("HGCBHInput", {"HGCalRecHit", "HGCHEBRecHits"});
  desc.add<edm::ParameterSetDescription>("HGCalConfig", psd4);

  // r/z windows
  desc.add<double>("nSigmasDeltaZ1", 5.0);      // in case beam spot is used for the matching
  desc.add<double>("deltaZ1WithVertex", 25.0);  // in case reco vertex is used for the matching
  desc.add<double>("z2MaxB", 0.09);
  desc.add<double>("r2MaxF", 0.15);
  desc.add<double>("rMaxI", 0.2);  // intermediate region SC in EB and 2nd hits in PXF

  // phi windows (dynamic)
  desc.add<double>("LowPtThreshold", 5.0);
  desc.add<double>("HighPtThreshold", 35.0);
  desc.add<double>("SizeWindowENeg", 0.675);
  desc.add<double>("DeltaPhi1Low", 0.23);
  desc.add<double>("DeltaPhi1High", 0.08);
  desc.add<double>("DeltaPhi2B", 0.008);
  desc.add<double>("DeltaPhi2F", 0.012);

  // phi windows (non dynamic, overwritten in case dynamic is selected)
  desc.add<double>("ePhiMin1", -0.125);
  desc.add<double>("ePhiMax1", 0.075);
  desc.add<double>("pPhiMin1", -0.075);
  desc.add<double>("pPhiMax1", 0.125);
  desc.add<double>("PhiMax2B", 0.002);
  desc.add<double>("PhiMax2F", 0.003);

  desc.add<edm::InputTag>("barrelSuperClusters",
                          {"particleFlowSuperClusterECAL", "particleFlowSuperClusterECALBarrel"});
  desc.add<edm::InputTag>("endcapSuperClusters",
                          {"particleFlowSuperClusterECAL", "particleFlowSuperClusterECALEndcapWithPreshower"});

  descriptions.add("ecalDrivenElectronSeeds", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronSeedProducer);
