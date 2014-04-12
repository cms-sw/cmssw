/* \class EgammaHLTBcHcalIsolationProducersRegional
 *
 * \author Matteo Sani (UCSD)
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTBcHcalIsolationProducersRegional.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

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

EgammaHLTBcHcalIsolationProducersRegional::EgammaHLTBcHcalIsolationProducersRegional(const edm::ParameterSet& config) {

  recoEcalCandidateProducer_ = consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"));
  caloTowerProducer_         = consumes<CaloTowerCollection>(config.getParameter<edm::InputTag>("caloTowerProducer"));

  doRhoCorrection_           = config.getParameter<bool>("doRhoCorrection");
  if (doRhoCorrection_)
    rhoProducer_               = consumes<double>(config.getParameter<edm::InputTag>("rhoProducer"));

  rhoMax_                    = config.getParameter<double>("rhoMax"); 
  rhoScale_                  = config.getParameter<double>("rhoScale"); 

  etMin_                     = config.getParameter<double>("etMin");  
  innerCone_                 = config.getParameter<double>("innerCone");
  outerCone_                 = config.getParameter<double>("outerCone");
  depth_                     = config.getParameter<int>("depth");
  doEtSum_                   = config.getParameter<bool>("doEtSum"); 
  effectiveAreaBarrel_       = config.getParameter<double>("effectiveAreaBarrel");
  effectiveAreaEndcap_       = config.getParameter<double>("effectiveAreaEndcap");

  useSingleTower_            = config.getParameter<bool>("useSingleTower");
  
  hcalCfg_.hOverEConeSize = outerCone_;
  hcalCfg_.useTowers = true;
  hcalCfg_.hcalTowers = caloTowerProducer_;
  hcalCfg_.hOverEPtMin = etMin_;

  hcalHelper_ = new ElectronHcalHelper(hcalCfg_);

  produces <reco::RecoEcalCandidateIsolationMap>(); 
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
  desc.add<double>(("effectiveAreaBarrel"), 0.021);
  desc.add<double>(("effectiveAreaEndcap"), 0.040);
  desc.add<bool>(("useSingleTower"), false);
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

  reco::RecoEcalCandidateIsolationMap isoMap;
  
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
      
      if (doRhoCorrection_) {
	if (fabs(recoEcalCandRef->superCluster()->eta()) < 1.442) 
	  isol = isol - rho*effectiveAreaBarrel_;
	else
	  isol = isol - rho*effectiveAreaEndcap_;
      }
    } else { //calcuate H for H/E
      if (useSingleTower_) 
	isol = hcalHelper_->hcalESumDepth1BehindClusters(towersBehindCluster) + hcalHelper_->hcalESumDepth2BehindClusters(towersBehindCluster);
      else
	isol = hcalHelper_->hcalESum(*(recoEcalCandRef->superCluster()));
    }

    isoMap.insert(recoEcalCandRef, isol);
  }
  
  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);
}
