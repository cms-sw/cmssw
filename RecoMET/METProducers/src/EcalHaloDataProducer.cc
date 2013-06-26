#include "RecoMET/METProducers/interface/EcalHaloDataProducer.h"

/*
  [class]:  EcalHaloDataProducer
  [authors]: R. Remington, The University of Florida
  [description]: See EcalHaloDataProducer.h
  [date]: October 15, 2009
*/

using namespace edm;
using namespace std;
using namespace reco;

EcalHaloDataProducer::EcalHaloDataProducer(const edm::ParameterSet& iConfig)
{
  //RecHit Level
  IT_EBRecHit    = iConfig.getParameter<edm::InputTag>("EBRecHitLabel");
  IT_EERecHit    = iConfig.getParameter<edm::InputTag>("EERecHitLabel");
  IT_ESRecHit    = iConfig.getParameter<edm::InputTag>("ESRecHitLabel");

  //Higher Level Reco 
  IT_SuperCluster = iConfig.getParameter<edm::InputTag>("SuperClusterLabel");
  IT_Photon = iConfig.getParameter<edm::InputTag>("PhotonLabel") ;

  // Shower Shape cuts for EcalAlgo

  RoundnessCut = iConfig.getParameter<double>("RoundnessCutParam");
  AngleCut = iConfig.getParameter<double>("AngleCutParam");

  EBRecHitEnergyThreshold = (float) iConfig.getParameter<double> ("EBRecHitEnergyThresholdParam");
  EERecHitEnergyThreshold = (float) iConfig.getParameter<double> ("EERecHitEnergyThresholdParam");
  ESRecHitEnergyThreshold = (float) iConfig.getParameter<double> ("ESRecHitEnergyThresholdParam");
  SumEcalEnergyThreshold = (float)iConfig.getParameter<double> ("SumEcalEnergyThresholdParam");
  NHitsEcalThreshold = iConfig.getParameter<int> ("NHitsEcalThresholdParam");
  
  RoundnessCut = iConfig.getParameter<double>("RoundnessCutParam");
  AngleCut = iConfig.getParameter<double>("AngleCutParam");

  produces<EcalHaloData>();
}

void EcalHaloDataProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  //Get CaloGeometry
  edm::ESHandle<CaloGeometry> TheCaloGeometry;
  iSetup.get<CaloGeometryRecord>().get(TheCaloGeometry);

  //Get  EB RecHits
  edm::Handle<EBRecHitCollection> TheEBRecHits;
  iEvent.getByLabel(IT_EBRecHit, TheEBRecHits);

  //Get EE RecHits
  edm::Handle<EERecHitCollection> TheEERecHits;
  iEvent.getByLabel(IT_EERecHit, TheEERecHits);

  //Get ES RecHits
  edm::Handle<ESRecHitCollection> TheESRecHits;
  iEvent.getByLabel(IT_ESRecHit, TheESRecHits);

  //Get ECAL Barrel SuperClusters                  
  edm::Handle<reco::SuperClusterCollection> TheSuperClusters;
  iEvent.getByLabel(IT_SuperCluster, TheSuperClusters);

  //Get Photons
  edm::Handle<reco::PhotonCollection> ThePhotons;
  iEvent.getByLabel(IT_Photon, ThePhotons);

  //Run the EcalHaloAlgo to reconstruct the EcalHaloData object 
  EcalHaloAlgo EcalAlgo;
  EcalAlgo.SetRoundnessCut(RoundnessCut);
  EcalAlgo.SetAngleCut(AngleCut);
  EcalAlgo.SetRecHitEnergyThresholds(EBRecHitEnergyThreshold, EERecHitEnergyThreshold, ESRecHitEnergyThreshold);
  EcalAlgo.SetPhiWedgeThresholds(SumEcalEnergyThreshold, NHitsEcalThreshold);
  
  if( TheCaloGeometry.isValid() && ThePhotons.isValid() && TheSuperClusters.isValid()  &&  TheEBRecHits.isValid() && TheEERecHits.isValid() && TheESRecHits.isValid() )
    {
      std::auto_ptr<EcalHaloData> EcalData( new EcalHaloData( EcalAlgo.Calculate(*TheCaloGeometry, ThePhotons, TheSuperClusters, TheEBRecHits, TheEERecHits, TheESRecHits)));
      iEvent.put( EcalData ) ; 
    }
  else 
    {
      std::auto_ptr<EcalHaloData> EcalData( new EcalHaloData() ) ;
      iEvent.put(EcalData); 
    }
  return;
}

EcalHaloDataProducer::~EcalHaloDataProducer(){}
