/* \class EgammaHLTBcHcalIsolationProducersRegional
 *
 * \author Matteo Sani (UCSD)
 *
 */

#include <iostream>
#include <vector>
#include <memory>

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTBcHcalIsolationProducersRegional.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

EgammaHLTBcHcalIsolationProducersRegional::EgammaHLTBcHcalIsolationProducersRegional(const edm::ParameterSet& config) :
  doEtSum_(                   config.getParameter<bool>("doEtSum") ),
  etMin_(                     config.getParameter<double>("etMin") ),
  innerCone_(                 config.getParameter<double>("innerCone") ),
  outerCone_(                 config.getParameter<double>("outerCone") ),
  depth_(                     config.getParameter<int>("depth") ),
  useSingleTower_(            config.getParameter<bool>("useSingleTower") ),
  doRhoCorrection_(           config.getParameter<bool>("doRhoCorrection") ),
  rhoScale_(                  config.getParameter<double>("rhoScale") ),
  rhoMax_(                    config.getParameter<double>("rhoMax") ),
  effectiveAreas_(            config.getParameter<std::vector<double> >("effectiveAreas") ),
  absEtaLowEdges_(            config.getParameter<std::vector<double> >("absEtaLowEdges") ),
  recoEcalCandidateProducer_( consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer")) ),
  caloTowerProducer_(         consumes<CaloTowerCollection>(config.getParameter<edm::InputTag>("caloTowerProducer")) ),
  rhoProducer_(               doRhoCorrection_ ? consumes<double>(config.getParameter<edm::InputTag>("rhoProducer")) : edm::EDGetTokenT<double>() )
{

  if (doRhoCorrection_) {
    if (absEtaLowEdges_.size() != effectiveAreas_.size())
      throw cms::Exception("IncompatibleVects") << "absEtaLowEdges and effectiveAreas should be of the same size. \n";

    if (absEtaLowEdges_.at(0) != 0.0)
      throw cms::Exception("IncompleteCoverage") << "absEtaLowEdges should start from 0. \n";

    for (unsigned int aIt = 0; aIt < absEtaLowEdges_.size() - 1; aIt++) {
      if ( !(absEtaLowEdges_.at( aIt ) < absEtaLowEdges_.at( aIt + 1 )) )
        throw cms::Exception("ImproperBinning") << "absEtaLowEdges entries should be in increasing order. \n";
    }
  }

  ElectronHcalHelper::Configuration hcalCfg;
  hcalCfg.hOverEConeSize    = outerCone_;
  hcalCfg.useTowers         = true;
  hcalCfg.hcalTowers        = caloTowerProducer_;
  hcalCfg.hOverEPtMin       = etMin_;
  hcalHelper_ = new ElectronHcalHelper(hcalCfg);

  produces<reco::RecoEcalCandidateIsolationMap>();
}

EgammaHLTBcHcalIsolationProducersRegional::~EgammaHLTBcHcalIsolationProducersRegional() {
  delete hcalHelper_;
}

void EgammaHLTBcHcalIsolationProducersRegional::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

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
  desc.add<std::vector<double> >("effectiveAreas", {0.079, 0.25}); // 2016 post-ichep sinEle default
  desc.add<std::vector<double> >("absEtaLowEdges", {0.0, 1.479}); // Barrel, Endcap
  descriptions.add(("hltEgammaHLTBcHcalIsolationProducersRegional"), desc);
}

void EgammaHLTBcHcalIsolationProducersRegional::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_, recoEcalCandHandle);

  edm::Handle<CaloTowerCollection> caloTowersHandle;
  iEvent.getByToken(caloTowerProducer_, caloTowersHandle);

  edm::Handle<double> rhoHandle;
  double rho = 0.0;

  if (doRhoCorrection_) {
    iEvent.getByToken(rhoProducer_, rhoHandle);
    rho = *(rhoHandle.product());
  }

  if (rho > rhoMax_)
    rho = rhoMax_;

  rho = rho*rhoScale_;

  hcalHelper_->checkSetup(iSetup);
  hcalHelper_->readEvent(iEvent);

  reco::RecoEcalCandidateIsolationMap isoMap(recoEcalCandHandle);

  for(unsigned int iRecoEcalCand=0; iRecoEcalCand <recoEcalCandHandle->size(); iRecoEcalCand++) {

    reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle, iRecoEcalCand);

    float isol = 0;

    std::vector<CaloTowerDetId> towersBehindCluster;

    if (useSingleTower_)
      towersBehindCluster = hcalHelper_->hcalTowersBehindClusters(*(recoEcalCandRef->superCluster()));

    if (doEtSum_) { //calculate hcal isolation excluding the towers behind the cluster which will be used for H for H/E
      EgammaTowerIsolation isolAlgo(outerCone_, innerCone_, etMin_, depth_, caloTowersHandle.product());
      if (useSingleTower_)
	isol = isolAlgo.getTowerEtSum(&(*recoEcalCandRef), &(towersBehindCluster)); // towersBehindCluster are excluded from the isolation sum
      else
	isol = isolAlgo.getTowerEtSum(&(*recoEcalCandRef));

    } else { //calcuate H for H/E
      if (useSingleTower_)
	isol = hcalHelper_->hcalESumDepth1BehindClusters(towersBehindCluster) + hcalHelper_->hcalESumDepth2BehindClusters(towersBehindCluster);
      else
	isol = hcalHelper_->hcalESum(*(recoEcalCandRef->superCluster()));

    }

    if (doRhoCorrection_) {
      int iEA = -1;
      auto scEta = std::abs(recoEcalCandRef->superCluster()->eta());
      for (int bIt = absEtaLowEdges_.size() - 1; bIt > -1; bIt--) {
        if ( scEta  > absEtaLowEdges_.at(bIt) ) {
          iEA = bIt;
          break;
        }
      }
        isol = isol - rho*effectiveAreas_.at(iEA);
    }

    isoMap.insert(recoEcalCandRef, isol);
  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);
}
