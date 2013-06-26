/** \class EgammaHLTHcalIsolationProducersRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTHcalIsolationProducersRegional.h"
#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolation.h"

// Framework
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

EgammaHLTHcalIsolationProducersRegional::EgammaHLTHcalIsolationProducersRegional(const edm::ParameterSet& config)
{
 // use configuration file to setup input/output collection names
  recoEcalCandidateProducer_ = config.getParameter<edm::InputTag>("recoEcalCandidateProducer");
  hbheRecHitProducer_        = config.getParameter<edm::InputTag>("hbheRecHitProducer");
  rhoProducer_               = config.getParameter<edm::InputTag>("rhoProducer");
  doRhoCorrection_           = config.getParameter<bool>("doRhoCorrection");
  rhoMax_                    = config.getParameter<double>("rhoMax"); 
  rhoScale_                  = config.getParameter<double>("rhoScale"); 
  
  double eMinHB              = config.getParameter<double>("eMinHB");
  double eMinHE              = config.getParameter<double>("eMinHE");
  double etMinHB             = config.getParameter<double>("etMinHB");  
  double etMinHE             = config.getParameter<double>("etMinHE");
  double innerCone           = config.getParameter<double>("innerCone");
  double outerCone           = config.getParameter<double>("outerCone");
  int depth                  = config.getParameter<int>("depth");
  doEtSum_                   = config.getParameter<bool>("doEtSum");
  effectiveAreaBarrel_       = config.getParameter<double>("effectiveAreaBarrel");
  effectiveAreaEndcap_       = config.getParameter<double>("effectiveAreaEndcap");
  isolAlgo_                  = new EgammaHLTHcalIsolation(eMinHB,eMinHE,etMinHB,etMinHE,innerCone,outerCone,depth);
 
  
  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();  
}

EgammaHLTHcalIsolationProducersRegional::~EgammaHLTHcalIsolationProducersRegional()
{
  delete isolAlgo_;
}



//
// member functions
//

// ------------ method called to produce the data  ------------
void
EgammaHLTHcalIsolationProducersRegional::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
  iEvent.getByLabel(recoEcalCandidateProducer_,recoEcalCandHandle);

  // Get the barrel hcal hits
  edm::Handle<HBHERecHitCollection>  hbheRecHitHandle;
  iEvent.getByLabel(hbheRecHitProducer_, hbheRecHitHandle);
  const HBHERecHitCollection* hbheRecHitCollection = hbheRecHitHandle.product();
  
  edm::ESHandle<HcalChannelQuality> hcalChStatus;
  iSetup.get<HcalChannelQualityRcd>().get(hcalChStatus);
  
  edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComp;
  iSetup.get<HcalSeverityLevelComputerRcd>().get(hcalSevLvlComp);

  edm::Handle<double> rhoHandle;
  double rho = 0.0;
  if (doRhoCorrection_) {
    iEvent.getByLabel(rhoProducer_, rhoHandle);
    rho = *(rhoHandle.product());
  }

  if (rho > rhoMax_)
    rho = rhoMax_;

  rho = rho*rhoScale_;

  edm::ESHandle<CaloGeometry> caloGeomHandle;
  iSetup.get<CaloGeometryRecord>().get(caloGeomHandle);
  const CaloGeometry* caloGeom = caloGeomHandle.product();
  
  reco::RecoEcalCandidateIsolationMap isoMap;
  
   
  for(reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoEcalCandHandle->begin(); iRecoEcalCand != recoEcalCandHandle->end(); iRecoEcalCand++){
    
    reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle,iRecoEcalCand -recoEcalCandHandle ->begin());
    
    float isol = 0;
    if(doEtSum_) {
      isol = isolAlgo_->getEtSum(recoEcalCandRef->superCluster()->eta(),
				 recoEcalCandRef->superCluster()->phi(),hbheRecHitCollection,caloGeom,
				 hcalSevLvlComp.product(),hcalChStatus.product());      
     
      if (doRhoCorrection_) {
	if (fabs(recoEcalCandRef->superCluster()->eta()) < 1.442) 
	  isol = isol - rho*effectiveAreaBarrel_;
	else
	  isol = isol - rho*effectiveAreaEndcap_;
      }
    } else {
      isol = isolAlgo_->getESum(recoEcalCandRef->superCluster()->eta(),recoEcalCandRef->superCluster()->phi(),
				hbheRecHitCollection,caloGeom,
				hcalSevLvlComp.product(),hcalChStatus.product());      
    }

    isoMap.insert(recoEcalCandRef, isol);   
  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);

}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTHcalIsolationProducersRegional);
