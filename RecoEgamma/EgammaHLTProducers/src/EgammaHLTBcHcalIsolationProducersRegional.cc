/* \class EgammaHLTBcHcalIsolationProducersRegional
 *
 * \author Matteo Sani (UCSD)
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTBcHcalIsolationProducersRegional.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
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

  recoEcalCandidateProducer_ = config.getParameter<edm::InputTag>("recoEcalCandidateProducer");
  caloTowerProducer_         = config.getParameter<edm::InputTag>("caloTowerProducer");
  rhoProducer_               = config.getParameter<edm::InputTag>("rhoProducer");
  doRhoCorrection_           = config.getParameter<bool>("doRhoCorrection");
  rhoMax_                    = config.getParameter<double>("rhoMax"); 
  rhoScale_                  = config.getParameter<double>("rhoScale"); 

  etMin_                     = config.getParameter<double>("etMin");  
  innerCone_                 = config.getParameter<double>("innerCone");
  outerCone_                 = config.getParameter<double>("outerCone");
  depth_                     = config.getParameter<int>("depth");
  doEtSum_                   = config.getParameter<bool>("doEtSum");
  effectiveAreaBarrel_       = config.getParameter<double>("effectiveAreaBarrel");
  effectiveAreaEndcap_       = config.getParameter<double>("effectiveAreaEndcap");
  
  hcalCfg.hOverEConeSize = 0.15;
  hcalCfg.useTowers = true;
  hcalCfg.hcalTowers = caloTowerProducer_;
  hcalCfg.hOverEPtMin = etMin_;

  hcalHelper = new ElectronHcalHelper(hcalCfg);

  produces <reco::RecoEcalCandidateIsolationMap>(); 
}

EgammaHLTBcHcalIsolationProducersRegional::~EgammaHLTBcHcalIsolationProducersRegional() {
  delete hcalHelper;
}

void EgammaHLTBcHcalIsolationProducersRegional::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
  iEvent.getByLabel(recoEcalCandidateProducer_, recoEcalCandHandle);

  edm::Handle<CaloTowerCollection> caloTowersHandle;
  iEvent.getByLabel(caloTowerProducer_, caloTowersHandle);

  edm::Handle<double> rhoHandle;
  double rho = 0.0;

  if (doRhoCorrection_) {
    iEvent.getByLabel(rhoProducer_, rhoHandle);
    rho = *(rhoHandle.product());
  }

  if (rho > rhoMax_)
    rho = rhoMax_;
  
  rho = rho*rhoScale_;
  
  hcalHelper->checkSetup(iSetup);
  hcalHelper->readEvent(iEvent);

  reco::RecoEcalCandidateIsolationMap isoMap;
  
  for(unsigned int iRecoEcalCand=0; iRecoEcalCand <recoEcalCandHandle->size(); iRecoEcalCand++) {
    
    reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle, iRecoEcalCand);
    
    float isol = 0;
    
    std::vector<CaloTowerDetId> towersToExclude = hcalHelper->hcalTowersBehindClusters(*(recoEcalCandRef->superCluster()));
    
    if (doEtSum_) {
      isoAlgo_ = new EgammaTowerIsolation(outerCone_, innerCone_, etMin_, depth_, caloTowersHandle.product());
      isol = isoAlgo_->getTowerEtSum(&(*recoEcalCandRef), &(towersToExclude));
      
      if (doRhoCorrection_) {
	if (fabs(recoEcalCandRef->superCluster()->eta()) < 1.442) 
	  isol = isol - rho*effectiveAreaBarrel_;
	else
	  isol = isol - rho*effectiveAreaEndcap_;
      }
    } else {
      isol = hcalHelper->hcalESumDepth1BehindClusters(towersToExclude) + hcalHelper->hcalESumDepth2BehindClusters(towersToExclude); 
    }

    isoMap.insert(recoEcalCandRef, isol);
  }
  
  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);
}
