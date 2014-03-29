#include "RecoMET/METProducers/interface/GlobalHaloDataProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

/*
  [class]:  GlobalHaloDataProducer
  [authors]: R. Remington, The University of Florida
  [description]: See GlobalHaloDataProducer.h
  [date]: October 15, 2009
*/

using namespace edm;
using namespace std;
using namespace reco;

GlobalHaloDataProducer::GlobalHaloDataProducer(const edm::ParameterSet& iConfig)
{
  //Higher Level Reco 
  IT_met = iConfig.getParameter<edm::InputTag>("metLabel");
  IT_CaloTower = iConfig.getParameter<edm::InputTag>("calotowerLabel");
  IT_CSCSegment = iConfig.getParameter<edm::InputTag>("CSCSegmentLabel");
  IT_CSCRecHit = iConfig.getParameter<edm::InputTag>("CSCRecHitLabel");

  //Halo Data from Sub-detectors
  IT_CSCHaloData = iConfig.getParameter<edm::InputTag>("CSCHaloDataLabel");
  IT_EcalHaloData = iConfig.getParameter<edm::InputTag> ("EcalHaloDataLabel");
  IT_HcalHaloData = iConfig.getParameter<edm::InputTag> ("HcalHaloDataLabel");

  EcalMinMatchingRadius = (float)iConfig.getParameter<double>("EcalMinMatchingRadiusParam");
  EcalMaxMatchingRadius = (float)iConfig.getParameter<double>("EcalMaxMatchingRadiusParam");
  HcalMinMatchingRadius = (float)iConfig.getParameter<double>("HcalMinMatchingRadiusParam");
  HcalMaxMatchingRadius = (float)iConfig.getParameter<double>("HcalMaxMatchingRadiusParam");
  CaloTowerEtThreshold  = (float)iConfig.getParameter<double>("CaloTowerEtThresholdParam");

  calotower_token_  = consumes<edm::View<Candidate> >(IT_CaloTower);
  calomet_token_    = consumes<reco::CaloMETCollection>(IT_met);
  cscsegment_token_ = consumes<CSCSegmentCollection>(IT_CSCSegment);
  cscrechit_token_  = consumes<CSCRecHit2DCollection>(IT_CSCRecHit);
  cschalo_token_    = consumes<CSCHaloData>(IT_CSCHaloData);
  ecalhalo_token_   = consumes<EcalHaloData>(IT_EcalHaloData);
  hcalhalo_token_   = consumes<HcalHaloData>(IT_HcalHaloData);

  produces<GlobalHaloData>();
}

void GlobalHaloDataProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  //Get CSC Geometry
  edm::ESHandle<CSCGeometry> TheCSCGeometry;
  iSetup.get<MuonGeometryRecord>().get(TheCSCGeometry);

  //Get Global Tracking Geometry
  edm::ESHandle<GlobalTrackingGeometry> TheGlobalTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(TheGlobalTrackingGeometry);

  //Get CaloGeometry
  edm::ESHandle<CaloGeometry> TheCaloGeometry;
  iSetup.get<CaloGeometryRecord>().get(TheCaloGeometry);

  //Get CaloTowers
  edm::Handle<edm::View<Candidate> > TheCaloTowers;
  //  iEvent.getByLabel(IT_CaloTower,TheCaloTowers);
  iEvent.getByToken(calotower_token_, TheCaloTowers);

  //Get MET
  edm::Handle< reco::CaloMETCollection > TheCaloMET;
  //  iEvent.getByLabel(IT_met, TheCaloMET);
  iEvent.getByToken(calomet_token_, TheCaloMET);

  //Get CSCSegments
  edm::Handle<CSCSegmentCollection> TheCSCSegments;
  //  iEvent.getByLabel(IT_CSCSegment, TheCSCSegments);
  iEvent.getByToken(cscsegment_token_, TheCSCSegments);

  //Get CSCRecHits
  edm::Handle<CSCRecHit2DCollection> TheCSCRecHits;
  //  iEvent.getByLabel(IT_CSCRecHit, TheCSCRecHits );
  iEvent.getByToken(cscrechit_token_, TheCSCRecHits);

  //Get CSCHaloData
  edm::Handle<reco::CSCHaloData> TheCSCHaloData;
  //  iEvent.getByLabel(IT_CSCHaloData, TheCSCHaloData );
  iEvent.getByToken(cschalo_token_, TheCSCHaloData);

  // Get EcalHaloData
  edm::Handle<reco::EcalHaloData> TheEcalHaloData;
  //  iEvent.getByLabel(IT_EcalHaloData, TheEcalHaloData );
  iEvent.getByToken(ecalhalo_token_, TheEcalHaloData);

  // Get HcalHaloData
  edm::Handle<reco::HcalHaloData> TheHcalHaloData;
  //  iEvent.getByLabel(IT_HcalHaloData, TheHcalHaloData );
  iEvent.getByToken(hcalhalo_token_, TheHcalHaloData);

  // Run the GlobalHaloAlgo to reconstruct the GlobalHaloData object 
  GlobalHaloAlgo GlobalAlgo;
  GlobalAlgo.SetEcalMatchingRadius(EcalMinMatchingRadius,EcalMaxMatchingRadius);
  GlobalAlgo.SetHcalMatchingRadius(HcalMinMatchingRadius,HcalMaxMatchingRadius);
  GlobalAlgo.SetCaloTowerEtThreshold(CaloTowerEtThreshold);
  //  GlobalHaloData GlobalData;

  if(TheCaloGeometry.isValid() && TheCaloMET.isValid() && TheCaloTowers.isValid() && TheCSCHaloData.isValid() && TheEcalHaloData.isValid() && TheHcalHaloData.isValid() )
    {
      std::auto_ptr<GlobalHaloData> GlobalData( new GlobalHaloData(GlobalAlgo.Calculate(*TheCaloGeometry, *TheCSCGeometry,  *(&TheCaloMET.product()->front()), TheCaloTowers, TheCSCSegments, TheCSCRecHits, *TheCSCHaloData.product(), *TheEcalHaloData.product(), *TheHcalHaloData.product() )) );
      iEvent.put(GlobalData);
    }
  else 
    {
      std::auto_ptr<GlobalHaloData> GlobalData( new GlobalHaloData() ) ;
      iEvent.put(GlobalData);
    }

  return;
}

GlobalHaloDataProducer::~GlobalHaloDataProducer(){}
