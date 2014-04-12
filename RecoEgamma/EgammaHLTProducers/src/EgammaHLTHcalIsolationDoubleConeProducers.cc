/** \class EgammaHLTHcalIsolationProducers
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 * $Id:
 *
 * mostly identical to EgammaHLTHcalIsolationRegionalProducers, but produces excludes  
 * Hcal energy in an exclusion cone around the eg candidate
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTHcalIsolationDoubleConeProducers.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

EgammaHLTHcalIsolationDoubleConeProducers::EgammaHLTHcalIsolationDoubleConeProducers(const edm::ParameterSet& config) : conf_(config)
{
 // use configuration file to setup input/output collection names
  recoEcalCandidateProducer_  = consumes<reco::RecoEcalCandidateCollection>(conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer"));
  hbRecHitProducer_           = consumes<HBHERecHitCollection>(conf_.getParameter<edm::InputTag>("hbRecHitProducer"));
  hfRecHitProducer_           = consumes<HFRecHitCollection>(conf_.getParameter<edm::InputTag>("hfRecHitProducer"));

  egHcalIsoPtMin_             = conf_.getParameter<double>("egHcalIsoPtMin");
  egHcalIsoConeSize_          = conf_.getParameter<double>("egHcalIsoConeSize");
  egHcalExclusion_            = conf_.getParameter<double>("egHcalExclusion");

  test_ = new EgammaHLTHcalIsolationDoubleCone(egHcalIsoPtMin_,egHcalIsoConeSize_,egHcalExclusion_);

  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();
}


EgammaHLTHcalIsolationDoubleConeProducers::~EgammaHLTHcalIsolationDoubleConeProducers(){delete test_;}

void EgammaHLTHcalIsolationDoubleConeProducers::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltL1SeededRecoEcalCandidate"));
  desc.add<edm::InputTag>(("hbRecHitProducer"), edm::InputTag("hltHbhereco"));
  desc.add<edm::InputTag>(("hfRecHitProducer"), edm::InputTag("hltHfreco"));
  desc.add<double>(("egHcalIsoPtMin"), 0.);
  desc.add<double>(("egHcalIsoConeSize"), 0.3);
  desc.add<double>(("egHcalExclusion"), 0.15);
  descriptions.add(("hltEgammaHLTHcalIsolationDoubleConeProducers"), desc);
}

void EgammaHLTHcalIsolationDoubleConeProducers::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_,recoecalcandHandle);
  
  // Get the barrel hcal hits
  edm::Handle<HBHERecHitCollection> hhitBarrelHandle;
  iEvent.getByToken(hbRecHitProducer_, hhitBarrelHandle);
  const HBHERecHitCollection* hcalhitBarrelCollection = hhitBarrelHandle.product();
  // Get the forward hcal hits
  edm::Handle<HFRecHitCollection> hhitEndcapHandle;
  iEvent.getByToken(hfRecHitProducer_, hhitEndcapHandle);
  const HFRecHitCollection* hcalhitEndcapCollection = hhitEndcapHandle.product();
  //Get Calo Geometry
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* caloGeom = pG.product();
  
  reco::RecoEcalCandidateIsolationMap isoMap;
  
  for(unsigned int iRecoEcalCand=0; iRecoEcalCand<recoecalcandHandle->size(); iRecoEcalCand++) {
    
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle, iRecoEcalCand);
    float isol =  test_->isolPtSum(&(*recoecalcandref), hcalhitBarrelCollection, hcalhitEndcapCollection, caloGeom);
    
    isoMap.insert(recoecalcandref, isol);
  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);

}
