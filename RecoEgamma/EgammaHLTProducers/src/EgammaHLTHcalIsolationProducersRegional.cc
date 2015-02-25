/** \class EgammaHLTHcalIsolationProducersRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTHcalIsolationProducersRegional.h"
#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolation.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"


#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

EgammaHLTHcalIsolationProducersRegional::EgammaHLTHcalIsolationProducersRegional(const edm::ParameterSet& config)
{
 // use configuration file to setup input/output collection names
  recoEcalCandidateProducer_ = consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"));
  hbheRecHitProducer_        = consumes<HBHERecHitCollection>(config.getParameter<edm::InputTag>("hbheRecHitProducer"));

  doRhoCorrection_           = config.getParameter<bool>("doRhoCorrection");
  if (doRhoCorrection_)
    rhoProducer_               = consumes<double>(config.getParameter<edm::InputTag>("rhoProducer"));

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

EgammaHLTHcalIsolationProducersRegional::~EgammaHLTHcalIsolationProducersRegional() {
  delete isolAlgo_;
}

void EgammaHLTHcalIsolationProducersRegional::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltL1SeededRecoEcalCandidate"));
  desc.add<edm::InputTag>(("hbheRecHitProducer"), edm::InputTag("hltHbhereco"));
  desc.add<edm::InputTag>(("rhoProducer"), edm::InputTag("fixedGridRhoFastjetAllCalo"));
  desc.add<bool>(("doRhoCorrection"), false);
  desc.add<double>(("rhoMax"), 9.9999999E7); 
  desc.add<double>(("rhoScale"), 1.0); 
  desc.add<double>(("eMinHB"), 0.7);
  desc.add<double>(("eMinHE"), 0.8);
  desc.add<double>(("etMinHB"), -1.0);  
  desc.add<double>(("etMinHE"), -1.0);
  desc.add<double>(("innerCone"), 0);
  desc.add<double>(("outerCone"), 0.15);
  desc.add<int>(("depth"),  -1);
  desc.add<bool>(("doEtSum"), false);
  desc.add<double>(("effectiveAreaBarrel"), 0.105);
  desc.add<double>(("effectiveAreaEndcap"), 0.170);
  descriptions.add(("hltEgammaHLTHcalIsolationProducersRegional"), desc);  
}

void EgammaHLTHcalIsolationProducersRegional::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_,recoEcalCandHandle);

  // Get the barrel hcal hits
  edm::Handle<HBHERecHitCollection>  hbheRecHitHandle;
  iEvent.getByToken(hbheRecHitProducer_, hbheRecHitHandle);
  const HBHERecHitCollection* hbheRecHitCollection = hbheRecHitHandle.product();
  
  edm::ESHandle<HcalChannelQuality> hcalChStatusHandle;    
  iSetup.get<HcalChannelQualityRcd>().get( "withTopo", hcalChStatusHandle );
  const HcalChannelQuality* hcalChStatus = hcalChStatusHandle.product();

  edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComp;
  iSetup.get<HcalSeverityLevelComputerRcd>().get(hcalSevLvlComp);

  edm::Handle<double> rhoHandle;
  double rho = 0.0;
  if (doRhoCorrection_) {
    iEvent.getByToken(rhoProducer_, rhoHandle);
    rho = *(rhoHandle.product());
  }

  if (rho > rhoMax_)
    rho = rhoMax_;

  rho = rho*rhoScale_;

  edm::ESHandle<CaloGeometry> caloGeomHandle;
  iSetup.get<CaloGeometryRecord>().get(caloGeomHandle);
  const CaloGeometry* caloGeom = caloGeomHandle.product();
  
  reco::RecoEcalCandidateIsolationMap isoMap(recoEcalCandHandle);
  
   
  for(reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoEcalCandHandle->begin(); iRecoEcalCand != recoEcalCandHandle->end(); iRecoEcalCand++){
    
    reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle,iRecoEcalCand -recoEcalCandHandle ->begin());
    
    float isol = 0;
    if(doEtSum_) {
      isol = isolAlgo_->getEtSum(recoEcalCandRef->superCluster()->eta(),
				 recoEcalCandRef->superCluster()->phi(),hbheRecHitCollection,caloGeom,
				 hcalSevLvlComp.product(),hcalChStatus);      
     
      if (doRhoCorrection_) {
	if (fabs(recoEcalCandRef->superCluster()->eta()) < 1.442) 
	  isol = isol - rho*effectiveAreaBarrel_;
	else
	  isol = isol - rho*effectiveAreaEndcap_;
      }
    } else {
      isol = isolAlgo_->getESum(recoEcalCandRef->superCluster()->eta(),recoEcalCandRef->superCluster()->phi(),
				hbheRecHitCollection,caloGeom,
				hcalSevLvlComp.product(),hcalChStatus);      
    }

    isoMap.insert(recoEcalCandRef, isol);   
  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);

}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTHcalIsolationProducersRegional);
