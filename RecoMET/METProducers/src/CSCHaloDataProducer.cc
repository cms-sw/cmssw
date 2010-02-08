#include "RecoMET/METProducers/interface/CSCHaloDataProducer.h"

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
  produces<CSCHaloData>();
}

void CSCHaloDataProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  //Get CSC Geometry
  edm::ESHandle<CSCGeometry> TheCSCGeometry;
  iSetup.get<MuonGeometryRecord>().get(TheCSCGeometry);

  //Get CSC Stand-Alone Muons from Cosmic Reconstruction 
  edm::Handle< reco::TrackCollection > TheCosmics;
  iEvent.getByLabel(IT_CosmicMuon, TheCosmics);
  
  //Get CSC Segments
  edm::Handle<CSCSegmentCollection> TheCSCSegments;
  iEvent.getByLabel(IT_CSCSegment, TheCSCSegments);

  //Get CSC RecHits
  Handle<CSCRecHit2DCollection> TheCSCRecHits;
  iEvent.getByLabel(IT_CSCRecHit, TheCSCRecHits);

  //Get L1MuGMT 
  edm::Handle < L1MuGMTReadoutCollection > TheL1GMTReadout ;
  iEvent.getByLabel (IT_L1MuGMTReadout, TheL1GMTReadout);

  //Get HLT Results                                                                                                                                                       
  edm::Handle<edm::TriggerResults> TheHLTResults;
  iEvent.getByLabel( IT_HLTResult , TheHLTResults);

  // Run The CSCHaloAlgo to reconstruct the CSCHaloData object
  // 
  /*
  if(TheCosmics.isValid() && TheCSCSegments.isValid() && TheCSCRecHits.isValid() && TheL1GMTReadout.isValid() && TheCSCGeometry.isValid() )
    {
      std::auto_ptr<CSCHaloData> TheCSCData(new CSCHaloData( CSCAlgo.Calculate(*TheCSCGeometry, TheCosmics, TheCSCSegments, TheCSCRecHits, TheL1GMTReadout) ) );
      // Put it in the event
      iEvent.put(TheCSCData);
    }
  else 
    {
      std::auto_ptr<CSCHaloData> TheCSCData(new CSCHaloData());
      // Put it in the event
      iEvent.put(TheCSCData);
    }
  */
  std::auto_ptr<CSCHaloData> TheCSCData(new CSCHaloData( CSCAlgo.Calculate(*TheCSCGeometry, TheCosmics, TheCSCSegments, TheCSCRecHits, TheL1GMTReadout, TheHLTResults) ) );
  // Put it in the event                                                                                                                                                
  iEvent.put(TheCSCData);
  return;

  
}

void CSCHaloDataProducer::beginJob(){return;}
void CSCHaloDataProducer::endJob(){return;}
void CSCHaloDataProducer::beginRun(edm::Run&, const edm::EventSetup&){return;}
void CSCHaloDataProducer::endRun(edm::Run&, const edm::EventSetup&){return;}
CSCHaloDataProducer::~CSCHaloDataProducer(){}
