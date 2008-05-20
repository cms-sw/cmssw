/** \class EgammaHLTHcalIsolationProducersRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTHcalIsolationProducersRegional.h"

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
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

EgammaHLTHcalIsolationProducersRegional::EgammaHLTHcalIsolationProducersRegional(const edm::ParameterSet& config) : conf_(config)
{
 // use configuration file to setup input/output collection names
  recoEcalCandidateProducer_               = conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer");

  hbRecHitProducer_           = conf_.getParameter<edm::InputTag>("hbRecHitProducer");
  hfRecHitProducer_           = conf_.getParameter<edm::InputTag>("hfRecHitProducer");

  egHcalIsoPtMin_               = conf_.getParameter<double>("egHcalIsoPtMin");
  egHcalIsoConeSize_            = conf_.getParameter<double>("egHcalIsoConeSize");

  test_ = new EgammaHLTHcalIsolation(egHcalIsoPtMin_,egHcalIsoConeSize_);


  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();
}


EgammaHLTHcalIsolationProducersRegional::~EgammaHLTHcalIsolationProducersRegional(){delete test_;}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EgammaHLTHcalIsolationProducersRegional::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByLabel(recoEcalCandidateProducer_,recoecalcandHandle);
  // Get the barrel hcal hits
  edm::Handle<HBHERecHitCollection> hhitBarrelHandle;
  iEvent.getByLabel(hbRecHitProducer_, hhitBarrelHandle);
  const HBHERecHitCollection* hcalhitBarrelCollection = hhitBarrelHandle.product();
  // Get the forward hcal hits
  edm::Handle<HFRecHitCollection> hhitEndcapHandle;
  iEvent.getByLabel(hfRecHitProducer_, hhitEndcapHandle);
  const HFRecHitCollection* hcalhitEndcapCollection = hhitEndcapHandle.product();
  //Get Calo Geometry
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* caloGeom = pG.product();
  
  reco::RecoEcalCandidateIsolationMap isoMap;
  
   
  for(reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoecalcandHandle->begin(); iRecoEcalCand != recoecalcandHandle->end(); iRecoEcalCand++){
    
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle,iRecoEcalCand -recoecalcandHandle ->begin());
    
    
    const reco::RecoCandidate *tempiRecoEcalCand = &(*recoecalcandref);
    float isol =  test_->isolPtSum(tempiRecoEcalCand,hcalhitBarrelCollection,hcalhitEndcapCollection,caloGeom);
    
    isoMap.insert(recoecalcandref, isol);
    
  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);

}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTHcalIsolationProducersRegional);
