#include "RecoMET/METProducers/interface/CSCHaloDataProducer.h"
#include "FWCore/Common/interface/TriggerNames.h"

/*
  [class]:  CSCHaloDataProducer
  [authors]: R. Remington, The University of Florida
  [description]: See CSCHaloDataProducer.h
  [date]: October 15, 2009
*/

using namespace edm;
using namespace std;
using namespace reco;

CSCHaloDataProducer::CSCHaloDataProducer(const edm::ParameterSet& iConfig)
{
  //Digi Level 
  IT_L1MuGMTReadout = iConfig.getParameter<edm::InputTag>("L1MuGMTReadoutLabel");

  //HLT Level
  IT_HLTResult    = iConfig.getParameter<edm::InputTag>("HLTResultLabel");
  CSCAlgo.vIT_HLTBit = iConfig.getParameter< std::vector< edm::InputTag> >("HLTBitLabel");
  
  //RecHit Level
  IT_CSCRecHit   = iConfig.getParameter<edm::InputTag>("CSCRecHitLabel");

  //Higher Level Reco 
  IT_CSCSegment = iConfig.getParameter<edm::InputTag>("CSCSegmentLabel");  
  IT_CosmicMuon = iConfig.getParameter<edm::InputTag>("CosmicMuonLabel"); 
  IT_Muon = iConfig.getParameter<edm::InputTag>("MuonLabel");
  IT_SA   = iConfig.getParameter<edm::InputTag>("SALabel"); 
  IT_ALCT = iConfig.getParameter<edm::InputTag>("ALCTDigiLabel"); 

  //Muon to Segment Matching
  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  TheService = new MuonServiceProxy(serviceParameters);
  edm::ParameterSet matchParameters = iConfig.getParameter<edm::ParameterSet>("MatchParameters");
  edm::ConsumesCollector iC = consumesCollector();
  TheMatcher = new MuonSegmentMatcher(matchParameters, TheService,iC);

  // Cosmic track selection parameters
  CSCAlgo.SetDetaThreshold( (float) iConfig.getParameter<double>("DetaParam"));
  CSCAlgo.SetDphiThreshold( (float) iConfig.getParameter<double>("DphiParam"));
  CSCAlgo.SetMinMaxInnerRadius( (float) iConfig.getParameter<double>("InnerRMinParam") ,  (float) iConfig.getParameter<double>("InnerRMaxParam") );
  CSCAlgo.SetMinMaxOuterRadius( (float) iConfig.getParameter<double>("OuterRMinParam"), (float) iConfig.getParameter<double>("OuterRMaxParam"));
  CSCAlgo.SetNormChi2Threshold( (float) iConfig.getParameter<double>("NormChi2Param") );
 
  // MLR
  CSCAlgo.SetMaxSegmentRDiff( (float) iConfig.getParameter<double>("MaxSegmentRDiff") );
  CSCAlgo.SetMaxSegmentPhiDiff( (float) iConfig.getParameter<double>("MaxSegmentPhiDiff") );
  CSCAlgo.SetMaxSegmentTheta( (float) iConfig.getParameter<double>("MaxSegmentTheta") );
  // End MLR

  CSCAlgo.SetMaxDtMuonSegment( (float) iConfig.getParameter<double>("MaxDtMuonSegment") );
  CSCAlgo.SetMaxFreeInverseBeta( (float) iConfig.getParameter<double>("MaxFreeInverseBeta") );
  CSCAlgo.SetExpectedBX( (short int) iConfig.getParameter<int>("ExpectedBX") );
  CSCAlgo.SetRecHitTime0( (float) iConfig.getParameter<double>("RecHitTime0") );
  CSCAlgo.SetRecHitTimeWindow( (float) iConfig.getParameter<double>("RecHitTimeWindow") );
  CSCAlgo.SetMinMaxOuterMomentumTheta( (float)iConfig.getParameter<double>("MinOuterMomentumTheta"), (float)iConfig.getParameter<double>("MaxOuterMomentumTheta") );
  CSCAlgo.SetMatchingDPhiThreshold( (float)iConfig.getParameter<double>("MatchingDPhiThreshold") );
  CSCAlgo.SetMatchingDEtaThreshold( (float)iConfig.getParameter<double>("MatchingDEtaThreshold") );
  CSCAlgo.SetMatchingDWireThreshold(iConfig.getParameter<int>("MatchingDWireThreshold") );

  produces<CSCHaloData>();
}

void CSCHaloDataProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  //Get CSC Geometry
  edm::ESHandle<CSCGeometry> TheCSCGeometry;
  iSetup.get<MuonGeometryRecord>().get(TheCSCGeometry);

  //Get Muons Collection from Cosmic Reconstruction 
  edm::Handle< reco::MuonCollection > TheCosmics;
  iEvent.getByLabel(IT_CosmicMuon, TheCosmics);
  
  //Get Muon Time Information from Cosmic Reconstruction
  edm::Handle<reco::MuonTimeExtraMap> TheCSCTimeMap;
  iEvent.getByLabel(IT_CosmicMuon.label(),"csc",TheCSCTimeMap);

 //Collision Muon Collection
  edm::Handle< reco::MuonCollection> TheMuons;
  iEvent.getByLabel(IT_Muon, TheMuons);

  //Get CSC Segments
  edm::Handle<CSCSegmentCollection> TheCSCSegments;
  iEvent.getByLabel(IT_CSCSegment, TheCSCSegments);

  //Get CSC RecHits
  Handle<CSCRecHit2DCollection> TheCSCRecHits;
  iEvent.getByLabel(IT_CSCRecHit, TheCSCRecHits);

  //Get L1MuGMT 
  edm::Handle < L1MuGMTReadoutCollection > TheL1GMTReadout ;
  iEvent.getByLabel (IT_L1MuGMTReadout, TheL1GMTReadout);

  //Get Chamber Anode Trigger Information
  edm::Handle<CSCALCTDigiCollection> TheALCTs;
  iEvent.getByLabel (IT_ALCT, TheALCTs);

  //Get HLT Results                                                                                                                                                       
  edm::Handle<edm::TriggerResults> TheHLTResults;
  iEvent.getByLabel( IT_HLTResult , TheHLTResults);

  const edm::TriggerNames * triggerNames = 0;
  if (TheHLTResults.isValid()) {
    triggerNames = &iEvent.triggerNames(*TheHLTResults);
  }

  std::auto_ptr<CSCHaloData> TheCSCData(new CSCHaloData( CSCAlgo.Calculate(*TheCSCGeometry, TheCosmics, TheCSCTimeMap, TheMuons, TheCSCSegments, TheCSCRecHits, TheL1GMTReadout, TheHLTResults, triggerNames, TheALCTs, TheMatcher, iEvent) ) );
  // Put it in the event                                                                                                                                                
  iEvent.put(TheCSCData);
  return;
}

CSCHaloDataProducer::~CSCHaloDataProducer(){}
