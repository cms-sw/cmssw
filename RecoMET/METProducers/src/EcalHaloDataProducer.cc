#include "RecoMET/METProducers/interface/EcalHaloDataProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

/*
  [class]:  EcalHaloDataProducer
  [authors]: R. Remington, The University of Florida
  [description]: See EcalHaloDataProducer.h
  [date]: October 15, 2009
*/

using namespace edm;
using namespace std;
using namespace reco;

EcalHaloDataProducer::EcalHaloDataProducer(const edm::ParameterSet& iConfig) : EcalAlgo(consumesCollector()) {
  //RecHit Level
  IT_EBRecHit = iConfig.getParameter<edm::InputTag>("EBRecHitLabel");
  IT_EERecHit = iConfig.getParameter<edm::InputTag>("EERecHitLabel");
  IT_ESRecHit = iConfig.getParameter<edm::InputTag>("ESRecHitLabel");
  IT_HBHERecHit = iConfig.getParameter<edm::InputTag>("HBHERecHitLabel");

  //Higher Level Reco
  IT_SuperCluster = iConfig.getParameter<edm::InputTag>("SuperClusterLabel");
  IT_Photon = iConfig.getParameter<edm::InputTag>("PhotonLabel");

  // Shower Shape cuts for EcalAlgo

  RoundnessCut = iConfig.getParameter<double>("RoundnessCutParam");
  AngleCut = iConfig.getParameter<double>("AngleCutParam");

  EBRecHitEnergyThreshold = (float)iConfig.getParameter<double>("EBRecHitEnergyThresholdParam");
  EERecHitEnergyThreshold = (float)iConfig.getParameter<double>("EERecHitEnergyThresholdParam");
  ESRecHitEnergyThreshold = (float)iConfig.getParameter<double>("ESRecHitEnergyThresholdParam");
  SumEcalEnergyThreshold = (float)iConfig.getParameter<double>("SumEcalEnergyThresholdParam");
  NHitsEcalThreshold = iConfig.getParameter<int>("NHitsEcalThresholdParam");

  ebrechit_token_ = consumes<EBRecHitCollection>(IT_EBRecHit);
  eerechit_token_ = consumes<EERecHitCollection>(IT_EERecHit);
  esrechit_token_ = consumes<ESRecHitCollection>(IT_ESRecHit);
  hbherechit_token_ = consumes<HBHERecHitCollection>(IT_HBHERecHit);
  supercluster_token_ = consumes<reco::SuperClusterCollection>(IT_SuperCluster);
  photon_token_ = consumes<reco::PhotonCollection>(IT_Photon);
  calogeometry_token_ = esConsumes<CaloGeometry, CaloGeometryRecord>();

  produces<EcalHaloData>();
}

void EcalHaloDataProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  //Get CaloGeometry
  edm::ESHandle<CaloGeometry> TheCaloGeometry = iSetup.getHandle(calogeometry_token_);

  //Get  EB RecHits
  edm::Handle<EBRecHitCollection> TheEBRecHits;
  //  iEvent.getByLabel(IT_EBRecHit, TheEBRecHits);
  iEvent.getByToken(ebrechit_token_, TheEBRecHits);

  //Get EE RecHits
  edm::Handle<EERecHitCollection> TheEERecHits;
  //  iEvent.getByLabel(IT_EERecHit, TheEERecHits);
  iEvent.getByToken(eerechit_token_, TheEERecHits);

  //Get ES RecHits
  edm::Handle<ESRecHitCollection> TheESRecHits;
  //  iEvent.getByLabel(IT_ESRecHit, TheESRecHits);
  iEvent.getByToken(esrechit_token_, TheESRecHits);

  //Get HBHE RecHits
  edm::Handle<HBHERecHitCollection> TheHBHERecHits;
  iEvent.getByToken(hbherechit_token_, TheHBHERecHits);

  //Get ECAL Barrel SuperClusters
  edm::Handle<reco::SuperClusterCollection> TheSuperClusters;
  //  iEvent.getByLabel(IT_SuperCluster, TheSuperClusters);
  iEvent.getByToken(supercluster_token_, TheSuperClusters);

  //Get Photons
  edm::Handle<reco::PhotonCollection> ThePhotons;
  //  iEvent.getByLabel(IT_Photon, ThePhotons);
  iEvent.getByToken(photon_token_, ThePhotons);

  //Run the EcalHaloAlgo to reconstruct the EcalHaloData object
  EcalAlgo.SetRoundnessCut(RoundnessCut);
  EcalAlgo.SetAngleCut(AngleCut);
  EcalAlgo.SetRecHitEnergyThresholds(EBRecHitEnergyThreshold, EERecHitEnergyThreshold, ESRecHitEnergyThreshold);
  EcalAlgo.SetPhiWedgeThresholds(SumEcalEnergyThreshold, NHitsEcalThreshold);

  iEvent.put(std::make_unique<EcalHaloData>(EcalAlgo.Calculate(
      *TheCaloGeometry, ThePhotons, TheSuperClusters, TheEBRecHits, TheEERecHits, TheESRecHits, TheHBHERecHits, iSetup)));

  return;
}

void EcalHaloDataProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // RecHit level
  desc.add<edm::InputTag>("EBRecHitLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("EERecHitLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<edm::InputTag>("ESRecHitLabel", edm::InputTag("ecalPreshowerRecHit", "EcalRecHitsES"));
  // Higher level Reco
  desc.add<edm::InputTag>("HBHERecHitLabel", edm::InputTag("hbhereco"));
  desc.add<edm::InputTag>("SuperClusterLabel", edm::InputTag("correctedHybridSuperClusters"));
  desc.add<edm::InputTag>("PhotonLabel", edm::InputTag(""));
  desc.add<double>("EBRecHitEnergyThresholdParam", 0.3);
  desc.add<double>("EERecHitEnergyThresholdParam", 0.3);
  desc.add<double>("ESRecHitEnergyThresholdParam", 0.3);
  desc.add<double>("SumEcalEnergyThresholdParam", 10);
  desc.add<int>("NHitsEcalThresholdParam", 4);
  // Shower Shape cut parameters (defaults need to be optimized)
  desc.add<double>("RoundnessCutParam", 0.41);
  desc.add<double>("AngleCutParam", 0.51);
  descriptions.addWithDefaultLabel(desc);
}
