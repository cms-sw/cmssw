// Class: EgammaHLTHcalVarProducerFromRecHit

/*

Author: Swagata Mukherjee

Date: August 2021

This class is similar to the existing class EgammaHLTBcHcalIsolationProducersRegional, 
but the new feature in this code is that the HCAL recHits are used instead of the 
calotowers which is expected to be phased out sometime in Run3.
The old class can also be used until calotowers stay. After that, one need to switch to this new one. 

As the old producer code, this one also produces either Hcal isolation or H for H/E depending if doEtSum=true or false.
H for H/E = either a single HCAL tower behind SC, or towers in a cone, and hcal isolation has these tower(s) excluded.
A rho correction can be applied

*/

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

class EgammaHLTHcalVarProducerFromRecHit : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTHcalVarProducerFromRecHit(const edm::ParameterSet &);

public:
  void beginRun(edm::Run const &, edm::EventSetup const &);
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const final;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const bool doEtSum_;
  const EgammaHcalIsolation::arrayHB eThresHB_;
  const EgammaHcalIsolation::arrayHB etThresHB_;
  const EgammaHcalIsolation::arrayHE eThresHE_;
  const EgammaHcalIsolation::arrayHE etThresHE_;
  const double innerCone_;
  const double outerCone_;
  const int depth_;
  const int maxSeverityHB_;
  const int maxSeverityHE_;
  const bool useSingleTower_;
  const bool doRhoCorrection_;
  const double rhoScale_;
  const double rhoMax_;
  const std::vector<double> effectiveAreas_;
  const std::vector<double> absEtaLowEdges_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<HBHERecHitCollection> hbheRecHitsTag_;
  const edm::EDGetTokenT<double> rhoProducer_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalTopologyToken_;
  const edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> hcalChannelQualityToken_;
  const edm::ESGetToken<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd> hcalSevLvlComputerToken_;
  const edm::ESGetToken<CaloTowerConstituentsMap, CaloGeometryRecord> caloTowerConstituentsMapToken_;
  const edm::EDPutTokenT<reco::RecoEcalCandidateIsolationMap> putToken_;

  //Get HCAL thresholds from GT
  edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> hcalCutsToken_;
  bool cutsFromDB;
  HcalPFCuts const *hcalCuts = nullptr;
};

EgammaHLTHcalVarProducerFromRecHit::EgammaHLTHcalVarProducerFromRecHit(const edm::ParameterSet &config)
    : doEtSum_(config.getParameter<bool>("doEtSum")),
      eThresHB_(config.getParameter<EgammaHcalIsolation::arrayHB>("eThresHB")),
      etThresHB_(config.getParameter<EgammaHcalIsolation::arrayHB>("etThresHB")),
      eThresHE_(config.getParameter<EgammaHcalIsolation::arrayHE>("eThresHE")),
      etThresHE_(config.getParameter<EgammaHcalIsolation::arrayHE>("etThresHE")),
      innerCone_(config.getParameter<double>("innerCone")),
      outerCone_(config.getParameter<double>("outerCone")),
      depth_(config.getParameter<int>("depth")),
      maxSeverityHB_(config.getParameter<int>("maxSeverityHB")),
      maxSeverityHE_(config.getParameter<int>("maxSeverityHE")),
      useSingleTower_(config.getParameter<bool>("useSingleTower")),
      doRhoCorrection_(config.getParameter<bool>("doRhoCorrection")),
      rhoScale_(config.getParameter<double>("rhoScale")),
      rhoMax_(config.getParameter<double>("rhoMax")),
      effectiveAreas_(config.getParameter<std::vector<double> >("effectiveAreas")),
      absEtaLowEdges_(config.getParameter<std::vector<double> >("absEtaLowEdges")),
      recoEcalCandidateProducer_(consumes(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
      hbheRecHitsTag_(consumes(config.getParameter<edm::InputTag>("hbheRecHitsTag"))),
      rhoProducer_(doRhoCorrection_ ? consumes<double>(config.getParameter<edm::InputTag>("rhoProducer"))
                                    : edm::EDGetTokenT<double>()),
      caloGeometryToken_{esConsumes()},
      hcalTopologyToken_{esConsumes()},
      hcalChannelQualityToken_{esConsumes(edm::ESInputTag("", "withTopo"))},
      hcalSevLvlComputerToken_{esConsumes()},
      caloTowerConstituentsMapToken_{esConsumes()},
      putToken_{produces<reco::RecoEcalCandidateIsolationMap>()},
      cutsFromDB(
          config.getParameter<bool>("usePFThresholdsFromDB")) {  //Retrieve HCAL PF thresholds - from config or from DB
  if (doRhoCorrection_) {
    if (absEtaLowEdges_.size() != effectiveAreas_.size()) {
      throw cms::Exception("IncompatibleVects") << "absEtaLowEdges and effectiveAreas should be of the same size. \n";
    }

    if (absEtaLowEdges_.at(0) != 0.0) {
      throw cms::Exception("IncompleteCoverage") << "absEtaLowEdges should start from 0. \n";
    }

    for (unsigned int aIt = 0; aIt < absEtaLowEdges_.size() - 1; aIt++) {
      if (!(absEtaLowEdges_.at(aIt) < absEtaLowEdges_.at(aIt + 1))) {
        throw cms::Exception("ImproperBinning") << "absEtaLowEdges entries should be in increasing order. \n";
      }
    }
  }

  if (cutsFromDB) {
    hcalCutsToken_ = esConsumes<HcalPFCuts, HcalPFCutsRcd, edm::Transition::BeginRun>(edm::ESInputTag("", "withTopo"));
  }
}

void EgammaHLTHcalVarProducerFromRecHit::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recoEcalCandidateProducer", edm::InputTag("hltRecoEcalCandidate"));
  desc.add<edm::InputTag>("rhoProducer", edm::InputTag("fixedGridRhoFastjetAllCalo"));
  desc.add<edm::InputTag>("hbheRecHitsTag", edm::InputTag("hltHbhereco"));
  desc.add<bool>("doRhoCorrection", false);
  desc.add<double>("rhoMax", 999999.);
  desc.add<double>(("rhoScale"), 1.0);
  //eThresHB/HE are from RecoParticleFlow/PFClusterProducer/python/particleFlowRecHitHBHE_cfi.py
  desc.add<std::vector<double> >("eThresHB", {0.1, 0.2, 0.3, 0.3});
  desc.add<std::vector<double> >("etThresHB", {0, 0, 0, 0});
  desc.add<std::vector<double> >("eThresHE", {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
  desc.add<std::vector<double> >("etThresHE", {0, 0, 0, 0, 0, 0, 0});
  desc.add<bool>("usePFThresholdsFromDB", true);
  desc.add<double>("innerCone", 0);
  desc.add<double>("outerCone", 0.14);
  desc.add<int>("depth", 0);
  desc.add<int>("maxSeverityHB", 9);
  desc.add<int>("maxSeverityHE", 9);
  desc.add<bool>("doEtSum", false);
  desc.add<bool>("useSingleTower", false);
  desc.add<std::vector<double> >("effectiveAreas", {0.079, 0.25});  // 2016 post-ichep sinEle default
  desc.add<std::vector<double> >("absEtaLowEdges", {0.0, 1.479});   // Barrel, Endcap
  descriptions.add("hltEgammaHLTHcalVarProducerFromRecHit", desc);
}

void EgammaHLTHcalVarProducerFromRecHit::beginRun(edm::Run const &run, edm::EventSetup const &iSetup) {
  if (cutsFromDB) {
    hcalCuts = &iSetup.getData(hcalCutsToken_);
  }
}

void EgammaHLTHcalVarProducerFromRecHit::produce(edm::StreamID,
                                                 edm::Event &iEvent,
                                                 const edm::EventSetup &iSetup) const {
  auto recoEcalCandHandle = iEvent.getHandle(recoEcalCandidateProducer_);

  double rho = 0.0;

  if (doRhoCorrection_) {
    rho = iEvent.get(rhoProducer_);
    if (rho > rhoMax_) {
      rho = rhoMax_;
    }
    rho = rho * rhoScale_;
  }

  reco::RecoEcalCandidateIsolationMap isoMap(recoEcalCandHandle);

  for (unsigned int iRecoEcalCand = 0; iRecoEcalCand < recoEcalCandHandle->size(); iRecoEcalCand++) {
    reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle, iRecoEcalCand);

    float isol = 0;
    EgammaHcalIsolation::InclusionRule external;
    EgammaHcalIsolation::InclusionRule internal;

    if (useSingleTower_) {
      if (!doEtSum_) {  //this is single tower based H/E
        external = EgammaHcalIsolation::InclusionRule::isBehindClusterSeed;
        internal = EgammaHcalIsolation::InclusionRule::withinConeAroundCluster;
      } else {  //this is cone-based HCAL isolation with single tower based footprint removal
        external = EgammaHcalIsolation::InclusionRule::withinConeAroundCluster;
        internal = EgammaHcalIsolation::InclusionRule::isBehindClusterSeed;
      }
    } else {  //useSingleTower_=False means H/E is cone-based
      external = EgammaHcalIsolation::InclusionRule::withinConeAroundCluster;
      internal = EgammaHcalIsolation::InclusionRule::withinConeAroundCluster;
    }

    EgammaHcalIsolation thisHcalVar_ = EgammaHcalIsolation(external,
                                                           outerCone_,
                                                           internal,
                                                           innerCone_,
                                                           eThresHB_,
                                                           etThresHB_,
                                                           maxSeverityHB_,
                                                           eThresHE_,
                                                           etThresHE_,
                                                           maxSeverityHE_,
                                                           iEvent.get(hbheRecHitsTag_),
                                                           iSetup.getData(caloGeometryToken_),
                                                           iSetup.getData(hcalTopologyToken_),
                                                           iSetup.getData(hcalChannelQualityToken_),
                                                           iSetup.getData(hcalSevLvlComputerToken_),
                                                           iSetup.getData(caloTowerConstituentsMapToken_));

    if (useSingleTower_) {
      if (doEtSum_) {  //this is cone-based HCAL isolation with single tower based footprint removal
        isol = thisHcalVar_.getHcalEtSumBc(recoEcalCandRef.get(), depth_, hcalCuts);  //depth=0 means all depths
      } else {                                                                        //this is single tower based H/E
        isol = thisHcalVar_.getHcalESumBc(recoEcalCandRef.get(), depth_, hcalCuts);   //depth=0 means all depths
      }
    } else {           //useSingleTower_=False means H/E is cone-based.
      if (doEtSum_) {  //hcal iso
        isol = thisHcalVar_.getHcalEtSum(recoEcalCandRef.get(), depth_, hcalCuts);  //depth=0 means all depths
      } else {  // doEtSum_=False means sum up energy, this is for H/E
        isol = thisHcalVar_.getHcalESum(recoEcalCandRef.get(), depth_, hcalCuts);  //depth=0 means all depths
      }
    }

    if (doRhoCorrection_) {
      int iEA = -1;
      auto scEta = std::abs(recoEcalCandRef->superCluster()->eta());
      for (int bIt = absEtaLowEdges_.size() - 1; bIt > -1; bIt--) {
        if (scEta > absEtaLowEdges_.at(bIt)) {
          iEA = bIt;
          break;
        }
      }
      isol = isol - rho * effectiveAreas_.at(iEA);
    }

    isoMap.insert(recoEcalCandRef, isol);
  }

  iEvent.emplace(putToken_, isoMap);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTHcalVarProducerFromRecHit);
