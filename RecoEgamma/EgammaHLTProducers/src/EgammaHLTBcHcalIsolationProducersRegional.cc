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
  doEtSum_                   = config.getParameter<bool>("doEtSum"); //this variable (which I cant change the name of) switches between hcal isolation and H for H/E
  effectiveAreaBarrel_       = config.getParameter<double>("effectiveAreaBarrel");
  effectiveAreaEndcap_       = config.getParameter<double>("effectiveAreaEndcap");
  
  hcalCfg_.hOverEConeSize = 0.15;
  hcalCfg_.useTowers = true;
  hcalCfg_.hcalTowers = caloTowerProducer_;
  hcalCfg_.hOverEPtMin = etMin_;

  hcalHelper_ = new ElectronHcalHelper(hcalCfg_);

  produces <reco::RecoEcalCandidateIsolationMap>(); 
}

EgammaHLTBcHcalIsolationProducersRegional::~EgammaHLTBcHcalIsolationProducersRegional() {
  delete hcalHelper_;
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
  
  hcalHelper_->checkSetup(iSetup);
  hcalHelper_->readEvent(iEvent);

  reco::RecoEcalCandidateIsolationMap isoMap;
  
  for(unsigned int iRecoEcalCand=0; iRecoEcalCand <recoEcalCandHandle->size(); iRecoEcalCand++) {
    
    reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle, iRecoEcalCand);
    
    float isol = 0;
    
    std::vector<CaloTowerDetId> towersBehindCluster = hcalHelper_->hcalTowersBehindClusters(*(recoEcalCandRef->superCluster()));
    
    if (doEtSum_) { //calculate hcal isolation excluding the towers behind the cluster which will be used for H for H/E
      EgammaTowerIsolation isolAlgo(outerCone_, innerCone_, etMin_, depth_, caloTowersHandle.product());
      isol = isolAlgo.getTowerEtSum(&(*recoEcalCandRef), &(towersBehindCluster)); // towersBehindCluster are excluded from the isolation sum
      
      if (doRhoCorrection_) {
	if (fabs(recoEcalCandRef->superCluster()->eta()) < 1.442) 
	  isol = isol - rho*effectiveAreaBarrel_;
	else
	  isol = isol - rho*effectiveAreaEndcap_;
      }
    } else { //calcuate H for H/E
      isol = hcalHelper_->hcalESumDepth1BehindClusters(towersBehindCluster) + hcalHelper_->hcalESumDepth2BehindClusters(towersBehindCluster); //towers beind the cluster are for H for H/E
    }

    isoMap.insert(recoEcalCandRef, isol);
  }
  
  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);
}
