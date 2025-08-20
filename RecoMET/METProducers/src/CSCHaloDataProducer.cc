#include "RecoMET/METProducers/interface/CSCHaloDataProducer.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

/*
  [class]:  CSCHaloDataProducer
  [authors]: R. Remington, The University of Florida
  [description]: See CSCHaloDataProducer.h
  [date]: October 15, 2009
*/

using namespace edm;
using namespace std;
using namespace reco;

CSCHaloDataProducer::CSCHaloDataProducer(const edm::ParameterSet& iConfig) : CSCAlgo(consumesCollector()) {
  //Digi Level
  IT_L1MuGMTReadout = iConfig.getParameter<edm::InputTag>("L1MuGMTReadoutLabel");

  //HLT Level
  IT_HLTResult = iConfig.getParameter<edm::InputTag>("HLTResultLabel");
  CSCAlgo.vIT_HLTBit = iConfig.getParameter<std::vector<edm::InputTag> >("HLTBitLabel");

  //RecHit Level
  IT_CSCRecHit = iConfig.getParameter<edm::InputTag>("CSCRecHitLabel");

  //Calo RecHit
  IT_HBHErh = iConfig.getParameter<edm::InputTag>("HBHErhLabel");
  IT_ECALBrh = iConfig.getParameter<edm::InputTag>("ECALBrhLabel");
  IT_ECALErh = iConfig.getParameter<edm::InputTag>("ECALErhLabel");
  //Higher Level Reco
  IT_CSCSegment = iConfig.getParameter<edm::InputTag>("CSCSegmentLabel");
  IT_CosmicMuon = iConfig.getParameter<edm::InputTag>("CosmicMuonLabel");
  IT_Muon = iConfig.getParameter<edm::InputTag>("MuonLabel");
  IT_SA = iConfig.getParameter<edm::InputTag>("SALabel");
  IT_ALCT = iConfig.getParameter<edm::InputTag>("ALCTDigiLabel");

  //Muon to Segment Matching
  edm::ParameterSet matchParameters = iConfig.getParameter<edm::ParameterSet>("MatchParameters");
  edm::ConsumesCollector iC = consumesCollector();
  TheMatcher = new MuonSegmentMatcher(matchParameters, iC);

  // Cosmic track selection parameters
  CSCAlgo.SetDetaThreshold((float)iConfig.getParameter<double>("DetaParam"));
  CSCAlgo.SetDphiThreshold((float)iConfig.getParameter<double>("DphiParam"));
  CSCAlgo.SetMinMaxInnerRadius((float)iConfig.getParameter<double>("InnerRMinParam"),
                               (float)iConfig.getParameter<double>("InnerRMaxParam"));
  CSCAlgo.SetMinMaxOuterRadius((float)iConfig.getParameter<double>("OuterRMinParam"),
                               (float)iConfig.getParameter<double>("OuterRMaxParam"));
  CSCAlgo.SetNormChi2Threshold((float)iConfig.getParameter<double>("NormChi2Param"));

  // MLR
  CSCAlgo.SetMaxSegmentRDiff((float)iConfig.getParameter<double>("MaxSegmentRDiff"));
  CSCAlgo.SetMaxSegmentPhiDiff((float)iConfig.getParameter<double>("MaxSegmentPhiDiff"));
  CSCAlgo.SetMaxSegmentTheta((float)iConfig.getParameter<double>("MaxSegmentTheta"));
  // End MLR

  CSCAlgo.SetMaxDtMuonSegment((float)iConfig.getParameter<double>("MaxDtMuonSegment"));
  CSCAlgo.SetMaxFreeInverseBeta((float)iConfig.getParameter<double>("MaxFreeInverseBeta"));
  CSCAlgo.SetExpectedBX((short int)iConfig.getParameter<int>("ExpectedBX"));
  CSCAlgo.SetRecHitTime0((float)iConfig.getParameter<double>("RecHitTime0"));
  CSCAlgo.SetRecHitTimeWindow((float)iConfig.getParameter<double>("RecHitTimeWindow"));
  CSCAlgo.SetMinMaxOuterMomentumTheta((float)iConfig.getParameter<double>("MinOuterMomentumTheta"),
                                      (float)iConfig.getParameter<double>("MaxOuterMomentumTheta"));
  CSCAlgo.SetMatchingDPhiThreshold((float)iConfig.getParameter<double>("MatchingDPhiThreshold"));
  CSCAlgo.SetMatchingDEtaThreshold((float)iConfig.getParameter<double>("MatchingDEtaThreshold"));
  CSCAlgo.SetMatchingDWireThreshold(iConfig.getParameter<int>("MatchingDWireThreshold"));

  cscGeometry_token = esConsumes<CSCGeometry, MuonGeometryRecord>();
  cosmicmuon_token_ = consumes<reco::MuonCollection>(IT_CosmicMuon);
  csctimemap_token_ = consumes<reco::MuonTimeExtraMap>(edm::InputTag(IT_CosmicMuon.label(), "csc"));
  muon_token_ = consumes<reco::MuonCollection>(IT_Muon);
  cscsegment_token_ = consumes<CSCSegmentCollection>(IT_CSCSegment);
  cscrechit_token_ = consumes<CSCRecHit2DCollection>(IT_CSCRecHit);
  cscalct_token_ = consumes<CSCALCTDigiCollection>(IT_ALCT);
  l1mugmtro_token_ = consumes<L1MuGMTReadoutCollection>(IT_L1MuGMTReadout);
  hbhereco_token_ = consumes<HBHERecHitCollection>(IT_HBHErh);
  EcalRecHitsEB_token_ = consumes<EcalRecHitCollection>(IT_ECALBrh);
  EcalRecHitsEE_token_ = consumes<EcalRecHitCollection>(IT_ECALErh);
  hltresult_token_ = consumes<edm::TriggerResults>(IT_HLTResult);

  produces<CSCHaloData>();
}

void CSCHaloDataProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  //Digi Level
  desc.add<edm::InputTag>("L1MuGMTReadoutLabel", edm::InputTag("gtDigis"));

  // HLT
  desc.add<edm::InputTag>("HLTResultLabel", edm::InputTag("TriggerResults::HLT"));
  desc.add<std::vector<edm::InputTag> >("HLTBitLabel",
                                        {edm::InputTag("HLT_CSCBeamHalo"),
                                         edm::InputTag("HLT_CSCBeamHaloOverlapRing1"),
                                         edm::InputTag("HLT_CSCBeamHaloOverlapRing2"),
                                         edm::InputTag("HLT_CSCBeamHaloRing2or3")});

  // RecHit Level
  desc.add<edm::InputTag>("CSCRecHitLabel", edm::InputTag("csc2DRecHits"));
  desc.add<edm::InputTag>("HBHErhLabel", edm::InputTag("hbhereco"));
  desc.add<edm::InputTag>("ECALBrhLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("ECALErhLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));

  //Higher Level Reco
  desc.add<edm::InputTag>("CSCSegmentLabel", edm::InputTag("cscSegments"));
  desc.add<edm::InputTag>("CosmicMuonLabel", edm::InputTag("muonsFromCosmics"));
  desc.add<edm::InputTag>("MuonLabel", edm::InputTag("muons"));
  desc.add<edm::InputTag>("SALabel", edm::InputTag("standAloneMuons"));
  desc.add<edm::InputTag>("ALCTDigiLabel", edm::InputTag("muonCSCDigis", "MuonCSCALCTDigi"))
      ->setComment("Chamber Level Trigger Primitive");

  {
    // Muon-Segment matching
    edm::ParameterSetDescription matchParametersPSet;
    MuonSegmentMatcher::fillPSetDescription(matchParametersPSet);
    desc.add<edm::ParameterSetDescription>("MatchParameters", matchParametersPSet);
  }

  desc.add<double>("DetaParam", 0.1);
  desc.add<double>("DphiParam", 1.00);
  desc.add<double>("InnerRMinParam", 0.);
  desc.add<double>("InnerRMaxParam", 99999.);
  desc.add<double>("OuterRMinParam", 0.);
  desc.add<double>("OuterRMaxParam", 99999.);
  desc.add<double>("NormChi2Param", 8.);

  // MLR
  // Default values for identifying CSCSegments that are halo-like
  desc.add<double>("MaxSegmentRDiff", 10.0);
  desc.add<double>("MaxSegmentPhiDiff", 0.1);
  desc.add<double>("MaxSegmentTheta", 0.7);
  // end MLR

  desc.add<double>("MaxDtMuonSegment", -10.0)
      ->setComment("if this is halo we expect ( T_outer_segment - T_inner_segment) to be less than MaxDtMuonSegment");
  desc.add<double>("MaxFreeInverseBeta", 0.0)
      ->setComment("if this is halo we expect free inverse beta to be less than MaxFreeInverseBeta");
  desc.add<int>("ExpectedBX", 6)->setComment("if this is Data, the expected collision bx will be 3 instead of 6");

  // The expected time of a collision recHit will be t = time_0 + time-of-flight
  // A recHit more than +/- time_window from collision timing will be declared "out-of-time"
  // recHit times are in [ns]
  desc.add<double>("RecHitTime0", 0.)->setComment("rechit time in ns");
  desc.add<double>("RecHitTimeWindow", 25.)->setComment("rechit time in ns");
  desc.add<double>("MinOuterMomentumTheta", .10);
  desc.add<double>("MaxOuterMomentumTheta", 3.0);
  desc.add<double>("MatchingDPhiThreshold", 0.18);
  desc.add<double>("MatchingDEtaThreshold", 0.4);
  desc.add<int>("MatchingDWireThreshold", 5);
  descriptions.addWithDefaultLabel(desc);
}

void CSCHaloDataProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  //Get CSC Geometry
  edm::ESHandle<CSCGeometry> TheCSCGeometry = iSetup.getHandle(cscGeometry_token);

  //Get Muons Collection from Cosmic Reconstruction
  edm::Handle<reco::MuonCollection> TheCosmics;
  //  iEvent.getByLabel(IT_CosmicMuon, TheCosmics);
  iEvent.getByToken(cosmicmuon_token_, TheCosmics);

  //Get Muon Time Information from Cosmic Reconstruction
  edm::Handle<reco::MuonTimeExtraMap> TheCSCTimeMap;
  //  iEvent.getByLabel(IT_CosmicMuon.label(),"csc",TheCSCTimeMap);
  iEvent.getByToken(csctimemap_token_, TheCSCTimeMap);

  //Collision Muon Collection
  edm::Handle<reco::MuonCollection> TheMuons;
  //  iEvent.getByLabel(IT_Muon, TheMuons);
  iEvent.getByToken(muon_token_, TheMuons);

  //Get CSC Segments
  edm::Handle<CSCSegmentCollection> TheCSCSegments;
  //  iEvent.getByLabel(IT_CSCSegment, TheCSCSegments);
  iEvent.getByToken(cscsegment_token_, TheCSCSegments);

  //Get CSC RecHits
  Handle<CSCRecHit2DCollection> TheCSCRecHits;
  //  iEvent.getByLabel(IT_CSCRecHit, TheCSCRecHits);
  iEvent.getByToken(cscrechit_token_, TheCSCRecHits);

  //Get L1MuGMT
  edm::Handle<L1MuGMTReadoutCollection> TheL1GMTReadout;
  //  iEvent.getByLabel (IT_L1MuGMTReadout, TheL1GMTReadout);
  iEvent.getByToken(l1mugmtro_token_, TheL1GMTReadout);

  //Get Chamber Anode Trigger Information
  edm::Handle<CSCALCTDigiCollection> TheALCTs;
  //  iEvent.getByLabel (IT_ALCT, TheALCTs);
  iEvent.getByToken(cscalct_token_, TheALCTs);

  //Calo rec hits
  Handle<HBHERecHitCollection> hbhehits;
  iEvent.getByToken(hbhereco_token_, hbhehits);
  Handle<EcalRecHitCollection> ecalebhits;
  iEvent.getByToken(EcalRecHitsEB_token_, ecalebhits);
  Handle<EcalRecHitCollection> ecaleehits;
  iEvent.getByToken(EcalRecHitsEE_token_, ecaleehits);

  //Get HLT Results
  edm::Handle<edm::TriggerResults> TheHLTResults;
  //  iEvent.getByLabel( IT_HLTResult , TheHLTResults);
  iEvent.getByToken(hltresult_token_, TheHLTResults);

  const edm::TriggerNames* triggerNames = nullptr;
  if (TheHLTResults.isValid()) {
    triggerNames = &iEvent.triggerNames(*TheHLTResults);
  }

  // Put it in the event
  iEvent.put(std::make_unique<CSCHaloData>(CSCAlgo.Calculate(*TheCSCGeometry,
                                                             TheCosmics,
                                                             TheCSCTimeMap,
                                                             TheMuons,
                                                             TheCSCSegments,
                                                             TheCSCRecHits,
                                                             TheL1GMTReadout,
                                                             hbhehits,
                                                             ecalebhits,
                                                             ecaleehits,
                                                             TheHLTResults,
                                                             triggerNames,
                                                             TheALCTs,
                                                             TheMatcher,
                                                             iEvent,
                                                             iSetup)));
  return;
}
