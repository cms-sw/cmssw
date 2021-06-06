// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTBcHcalIsolationProducersRegional
//
// Original Author:  Matteo Sani (UCSD)
//         Created:  Thu Nov 24 11:38:00 CEST 2011
//

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//this class produces either Hcal isolation or H for H/E  depending if doEtSum=true or false
//H for H/E = towers behind SC, hcal isolation has these towers excluded
//a rho correction can be applied

class EgammaHLTBcHcalIsolationProducersRegional : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTBcHcalIsolationProducersRegional(const edm::ParameterSet &);

public:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const final;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const bool doEtSum_;
  const double etMin_;
  const double innerCone_;
  const double outerCone_;
  const int depth_;
  const bool useSingleTower_;

  const bool doRhoCorrection_;
  const double rhoScale_;
  const double rhoMax_;
  const std::vector<double> effectiveAreas_;
  const std::vector<double> absEtaLowEdges_;

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<CaloTowerCollection> caloTowerProducer_;
  const edm::EDGetTokenT<double> rhoProducer_;

  const edm::ESGetToken<CaloTowerConstituentsMap, CaloGeometryRecord> caloTowerConstituentsMapToken_;

  const edm::EDPutTokenT<reco::RecoEcalCandidateIsolationMap> putToken_;
};

EgammaHLTBcHcalIsolationProducersRegional::EgammaHLTBcHcalIsolationProducersRegional(const edm::ParameterSet &config)
    : doEtSum_(config.getParameter<bool>("doEtSum")),
      etMin_(config.getParameter<double>("etMin")),
      innerCone_(config.getParameter<double>("innerCone")),
      outerCone_(config.getParameter<double>("outerCone")),
      depth_(config.getParameter<int>("depth")),
      useSingleTower_(config.getParameter<bool>("useSingleTower")),
      doRhoCorrection_(config.getParameter<bool>("doRhoCorrection")),
      rhoScale_(config.getParameter<double>("rhoScale")),
      rhoMax_(config.getParameter<double>("rhoMax")),
      effectiveAreas_(config.getParameter<std::vector<double> >("effectiveAreas")),
      absEtaLowEdges_(config.getParameter<std::vector<double> >("absEtaLowEdges")),
      recoEcalCandidateProducer_(consumes(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
      caloTowerProducer_(consumes(config.getParameter<edm::InputTag>("caloTowerProducer"))),
      rhoProducer_(doRhoCorrection_ ? consumes<double>(config.getParameter<edm::InputTag>("rhoProducer"))
                                    : edm::EDGetTokenT<double>()),
      caloTowerConstituentsMapToken_{esConsumes()},
      putToken_{produces<reco::RecoEcalCandidateIsolationMap>()} {
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
}

void EgammaHLTBcHcalIsolationProducersRegional::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltRecoEcalCandidate"));
  desc.add<edm::InputTag>(("caloTowerProducer"), edm::InputTag("hltTowerMakerForAll"));
  desc.add<edm::InputTag>(("rhoProducer"), edm::InputTag("fixedGridRhoFastjetAllCalo"));
  desc.add<bool>(("doRhoCorrection"), false);
  desc.add<double>(("rhoMax"), 999999.);
  desc.add<double>(("rhoScale"), 1.0);
  desc.add<double>(("etMin"), -1.0);
  desc.add<double>(("innerCone"), 0);
  desc.add<double>(("outerCone"), 0.15);
  desc.add<int>(("depth"), -1);
  desc.add<bool>(("doEtSum"), false);
  desc.add<bool>(("useSingleTower"), false);
  desc.add<std::vector<double> >("effectiveAreas", {0.079, 0.25});  // 2016 post-ichep sinEle default
  desc.add<std::vector<double> >("absEtaLowEdges", {0.0, 1.479});   // Barrel, Endcap
  descriptions.add(("hltEgammaHLTBcHcalIsolationProducersRegional"), desc);
}

void EgammaHLTBcHcalIsolationProducersRegional::produce(edm::StreamID,
                                                        edm::Event &iEvent,
                                                        const edm::EventSetup &iSetup) const {
  // Get the HLT filtered objects
  auto recoEcalCandHandle = iEvent.getHandle(recoEcalCandidateProducer_);

  double rho = 0.0;

  if (doRhoCorrection_) {
    rho = iEvent.get(rhoProducer_);
    if (rho > rhoMax_) {
      rho = rhoMax_;
    }
    rho = rho * rhoScale_;
  }

  auto const &caloTowers = iEvent.get(caloTowerProducer_);
  auto const &ctmaph = iSetup.getData(caloTowerConstituentsMapToken_);

  const EgammaTowerIsolation towerIso1(outerCone_, 0., etMin_, 1, &caloTowers);
  const EgammaTowerIsolation towerIso2(outerCone_, 0., etMin_, 2, &caloTowers);

  reco::RecoEcalCandidateIsolationMap isoMap(recoEcalCandHandle);

  for (unsigned int iRecoEcalCand = 0; iRecoEcalCand < recoEcalCandHandle->size(); iRecoEcalCand++) {
    reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle, iRecoEcalCand);

    float isol = 0;

    auto towersBehindCluster =
        useSingleTower_ ? egamma::towersOf(*(recoEcalCandRef->superCluster()), ctmaph) : std::vector<CaloTowerDetId>{};

    if (doEtSum_) {  //calculate hcal isolation excluding the towers behind the cluster which will be used for H for H/E
      const EgammaTowerIsolation isolAlgo(outerCone_, innerCone_, etMin_, depth_, &caloTowers);
      if (useSingleTower_) {
        // towersBehindCluster are excluded from the isolation sum
        isol = isolAlgo.getTowerEtSum(recoEcalCandRef.get(), &towersBehindCluster);
      } else {
        isol = isolAlgo.getTowerEtSum(recoEcalCandRef.get());
      }

    } else {  //calcuate H for H/E
      if (useSingleTower_)
        isol = egamma::depth1HcalESum(towersBehindCluster, caloTowers) +
               egamma::depth2HcalESum(towersBehindCluster, caloTowers);
      else {
        auto const &sc = recoEcalCandRef->superCluster().get();
        isol = towerIso1.getTowerESum(sc) + towerIso2.getTowerESum(sc);
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
DEFINE_FWK_MODULE(EgammaHLTBcHcalIsolationProducersRegional);
