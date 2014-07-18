#include "RecoMET/METProducers/interface/HcalHaloDataProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

/*
  [class]:  HcalHaloDataProducer
  [authors]: R. Remington, The University of Florida
  [description]: See HcalHaloDataProducer.h
  [date]: October 15, 2009
*/

using namespace edm;
using namespace std;
using namespace reco;

HcalHaloDataProducer::HcalHaloDataProducer(const edm::ParameterSet& iConfig)
{
  //RecHit Level
  IT_HBHERecHit  = iConfig.getParameter<edm::InputTag>("HBHERecHitLabel");
  IT_HFRecHit    = iConfig.getParameter<edm::InputTag>("HFRecHitLabel");
  IT_HORecHit    = iConfig.getParameter<edm::InputTag>("HORecHitLabel");

  HBRecHitEnergyThreshold = (float)iConfig.getParameter<double>("HBRecHitEnergyThresholdParam");
  HERecHitEnergyThreshold = (float)iConfig.getParameter<double>("HERecHitEnergyThresholdParam");
  SumHcalEnergyThreshold = (float) iConfig.getParameter<double>("SumHcalEnergyThresholdParam");
  NHitsHcalThreshold =  iConfig.getParameter<int>("NHitsHcalThresholdParam");

  hbherechit_token_ = consumes<HBHERecHitCollection>(IT_HBHERecHit);
  hfrechit_token_ = consumes<HFRecHitCollection>(IT_HFRecHit);

  produces<HcalHaloData>();
}

void HcalHaloDataProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  //Get CaloGeometry
  edm::ESHandle<CaloGeometry> TheCaloGeometry;
  iSetup.get<CaloGeometryRecord>().get(TheCaloGeometry);
  
  //Get HB/HE RecHits
  edm::Handle<HBHERecHitCollection> TheHBHERecHits;
  //  iEvent.getByLabel(IT_HBHERecHit, TheHBHERecHits);
  iEvent.getByToken(hbherechit_token_, TheHBHERecHits);

  //Get HF RecHits
  edm::Handle<HFRecHitCollection> TheHFRecHits;
  //  iEvent.getByLabel(IT_HFRecHit, TheHFRecHits);
  iEvent.getByToken(hfrechit_token_, TheHFRecHits);

  // Run the HcalHaloAlgo to reconstruct the HcalHaloData object
  HcalHaloAlgo HcalAlgo;
  HcalAlgo.SetRecHitEnergyThresholds( HBRecHitEnergyThreshold, HERecHitEnergyThreshold );
  HcalAlgo.SetPhiWedgeThresholds( SumHcalEnergyThreshold, NHitsHcalThreshold );

  HcalHaloData HcalData;
  if( TheCaloGeometry.isValid() && TheHBHERecHits.isValid() )
    {
      std::auto_ptr<HcalHaloData> HcalData( new HcalHaloData( HcalAlgo.Calculate(*TheCaloGeometry, TheHBHERecHits)  ) ) ;
      iEvent.put ( HcalData ) ;
    }
  else 
    {
      std::auto_ptr<HcalHaloData> HcalData( new HcalHaloData() ) ;
      iEvent.put( HcalData ) ;
    }
  return;
}

HcalHaloDataProducer::~HcalHaloDataProducer(){}
