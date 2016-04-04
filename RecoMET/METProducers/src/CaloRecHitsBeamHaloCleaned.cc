// -*- C++ -*-
//
// Package:    RecoMET/METProducers
// Class:      CaloRecHitsBeamHaloCleaned
// 
/**\class CaloRecHitsBeamHaloCleaned CaloRecHitsBeamHaloCleaned.cc RecoMET/METProducers/plugins/CaloRecHitsBeamHaloCleaned.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Laurent
//         Created:  Tue, 09 Feb 2016 13:09:37 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include <vector>
#include <iostream>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/METReco/interface/CSCHaloData.h"
#include "RecoMET/METAlgorithms/interface/CSCHaloAlgo.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"


class CaloRecHitsBeamHaloCleaned : public edm::stream::EDProducer<> {
   public:
      explicit CaloRecHitsBeamHaloCleaned(const edm::ParameterSet&);
      ~CaloRecHitsBeamHaloCleaned();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
  virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

  edm::EDGetTokenT<reco::TrackCollection > trk_token; 
  edm::EDGetTokenT<EcalRecHitCollection> ecalebhits_token;
  edm::EDGetTokenT<EcalRecHitCollection> ecaleehits_token;
  edm::EDGetTokenT<HBHERecHitCollection> hbhehits_token; 
  edm::EDGetTokenT<reco::MuonCollection> cosmicmuon_token_;
  edm::EDGetTokenT<reco::MuonTimeExtraMap> csctimemap_token_;
  edm::EDGetTokenT<reco::MuonCollection> muon_token_;
  edm::EDGetTokenT<CSCSegmentCollection> cscsegment_token_;
  edm::EDGetTokenT<CSCRecHit2DCollection> cscrechit_token_;
  edm::EDGetTokenT<CSCALCTDigiCollection> cscalct_token_;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> l1mugmtro_token_;
  edm::EDGetTokenT<edm::TriggerResults> hltresult_token_;

  //Input tags
  edm::InputTag IT_L1MuGMTReadout;
  edm::InputTag IT_ALCT;
  edm::InputTag IT_HLTResult;
  std::vector< edm::InputTag > vIT_HLTBit  ;
  edm::InputTag IT_CSCRecHit;
  edm::InputTag IT_CosmicMuon;
  edm::InputTag IT_CSCSegment;
  edm::InputTag IT_Muon;
  edm::InputTag IT_EBRecHits;
  edm::InputTag IT_EERecHits;
  edm::InputTag IT_HBHERecHits;


  CSCHaloAlgo CSCAlgo;
  MuonSegmentMatcher * TheMatcher;

  
  bool ishlt;
};

//
// constructors and destructor
//
CaloRecHitsBeamHaloCleaned::CaloRecHitsBeamHaloCleaned(const edm::ParameterSet& iConfig)
{

  ishlt = iConfig.getUntrackedParameter< bool> ("IsHLT",false);

  CSCAlgo.vIT_HLTBit = iConfig.getParameter< std::vector< edm::InputTag> >("HLTBitLabel");

  produces<EcalRecHitCollection>("EcalRecHitsEB");
  produces<EcalRecHitCollection>("EcalRecHitsEE");
  produces<HBHERecHitCollection>();


  CSCAlgo.vIT_HLTBit = iConfig.getParameter< std::vector< edm::InputTag> >("HLTBitLabel");  
  //Digi Level
  IT_L1MuGMTReadout = iConfig.getParameter<edm::InputTag>("L1MuGMTReadoutLabel");
  //HLT Level
  IT_HLTResult    = iConfig.getParameter<edm::InputTag>("HLTResultLabel");
  //RecHit Level
  IT_CSCRecHit   = iConfig.getParameter<edm::InputTag>("CSCRecHitLabel");
  //Higher Level Reco
  IT_CSCSegment = iConfig.getParameter<edm::InputTag>("CSCSegmentLabel");
  IT_CosmicMuon = iConfig.getParameter<edm::InputTag>("CosmicMuonLabel");
  IT_Muon = iConfig.getParameter<edm::InputTag>("MuonLabel");
  IT_ALCT = iConfig.getParameter<edm::InputTag>("ALCTDigiLabel");
  IT_EBRecHits = iConfig.getParameter<edm::InputTag>("EBRecHitsLabel");
  IT_EERecHits = iConfig.getParameter<edm::InputTag>("EERecHitsLabel");
  IT_HBHERecHits = iConfig.getParameter<edm::InputTag>("HBHERecHitsLabel");


  //Muon to Segment Matching
  edm::ParameterSet matchParameters = iConfig.getParameter<edm::ParameterSet>("MatchParameters");
  edm::ConsumesCollector iC = consumesCollector();
  TheMatcher = new MuonSegmentMatcher(matchParameters, iC);

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
   
  cosmicmuon_token_ = consumes<reco::MuonCollection>(IT_CosmicMuon);
  csctimemap_token_ = consumes<reco::MuonTimeExtraMap>(edm::InputTag(IT_CosmicMuon.label(), "csc"));
  muon_token_       = consumes<reco::MuonCollection>(IT_Muon);
  cscsegment_token_ = consumes<CSCSegmentCollection>(IT_CSCSegment);
  cscrechit_token_  = consumes<CSCRecHit2DCollection>(IT_CSCRecHit);
  cscalct_token_    = consumes<CSCALCTDigiCollection>(IT_ALCT);
  l1mugmtro_token_  = consumes<L1MuGMTReadoutCollection>(IT_L1MuGMTReadout);
  hltresult_token_  = consumes<edm::TriggerResults>(IT_HLTResult);

  ecalebhits_token= consumes<EcalRecHitCollection>(IT_EBRecHits);
  ecaleehits_token= consumes<EcalRecHitCollection>(IT_EERecHits);
  hbhehits_token= consumes<HBHERecHitCollection>(IT_HBHERecHits);


}


CaloRecHitsBeamHaloCleaned::~CaloRecHitsBeamHaloCleaned()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CaloRecHitsBeamHaloCleaned::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco; 
   using namespace std;


   Handle<EcalRecHitCollection> ebrhitsuncleaned;
   iEvent.getByToken(ecalebhits_token, ebrhitsuncleaned );

   Handle<EcalRecHitCollection> eerhitsuncleaned;
   iEvent.getByToken(ecaleehits_token, eerhitsuncleaned );

   Handle<HBHERecHitCollection> hbherhitsuncleaned;
   iEvent.getByToken(hbhehits_token, hbherhitsuncleaned );

   edm::ESHandle<CSCGeometry> TheCSCGeometry;
   iSetup.get<MuonGeometryRecord>().get(TheCSCGeometry);

   edm::Handle< reco::MuonCollection > TheCosmics;
   iEvent.getByToken(cosmicmuon_token_, TheCosmics);

   edm::Handle<reco::MuonTimeExtraMap> TheCSCTimeMap;
   iEvent.getByToken(csctimemap_token_, TheCSCTimeMap);

   edm::Handle< reco::MuonCollection> TheMuons;
   iEvent.getByToken(muon_token_, TheMuons);

   edm::Handle<CSCSegmentCollection> TheCSCSegments;
   iEvent.getByToken(cscsegment_token_, TheCSCSegments);

   edm::Handle<CSCRecHit2DCollection> TheCSCRecHits;
   iEvent.getByToken(cscrechit_token_, TheCSCRecHits);

   edm::Handle < L1MuGMTReadoutCollection > TheL1GMTReadout ;
   iEvent.getByToken(l1mugmtro_token_, TheL1GMTReadout);

   edm::Handle<edm::TriggerResults> TheHLTResults;
   iEvent.getByToken(hltresult_token_, TheHLTResults);
   
   const edm::TriggerNames * triggerNames = 0;
   if (TheHLTResults.isValid()) {
     triggerNames = &iEvent.triggerNames(*TheHLTResults);
   }
   
   edm::Handle<CSCALCTDigiCollection> TheALCTs;
   iEvent.getByToken(cscalct_token_, TheALCTs);
   
   //CSCHaloAlgo::Calculate returns a CSCHaloData object containing (amongst other things) the refs to rechits associated to a beam halo objectxs
   std::auto_ptr<CSCHaloData> TheCSCHaloData(new CSCHaloData( CSCAlgo.Calculate(*TheCSCGeometry, TheCosmics, TheCSCTimeMap, TheMuons, TheCSCSegments, TheCSCRecHits, TheL1GMTReadout, hbherhitsuncleaned, ebrhitsuncleaned, eerhitsuncleaned ,TheHLTResults, triggerNames, TheALCTs, TheMatcher, iEvent, iSetup,ishlt) ) );
   const CSCHaloData TheSummaryHalo = (*TheCSCHaloData );
   
   if( (TheSummaryHalo.GetEBRechits()).size() >0 || (TheSummaryHalo.GetEERechits()).size() >0 || (TheSummaryHalo.GetHBHERechits()).size() >0   ){
     cout << 
       "BH calohits found " <<
       (TheSummaryHalo.GetEBRechits()).size()<<", "<<
       (TheSummaryHalo.GetEERechits()).size()<< ", "<<
       (TheSummaryHalo.GetHBHERechits()).size()<<", "<<
       endl;
   }

   //Cleaning of the various rechits collections:

   //  EcalRecHit EB
   auto_ptr<EcalRecHitCollection> ebrhitscleaned(new EcalRecHitCollection()); 
   for(unsigned int i = 0;  i < ebrhitsuncleaned->size(); i++){
     const EcalRecHit & rhit = (*ebrhitsuncleaned)[i];
     bool isclean(true);
     edm::RefVector<EcalRecHitCollection> refbeamhalorechits =  TheSummaryHalo.GetEBRechits();
     for(unsigned int j = 0; j <refbeamhalorechits.size() ; j++){
       const EcalRecHit &rhitbeamhalo = *(refbeamhalorechits)[j];
       if( rhit.detid() == rhitbeamhalo.detid() ) { 
	 isclean  = false;
	 break;
       }
     }
     if(isclean) ebrhitscleaned->push_back(rhit);
   }
   
   //  EcalRecHit EE
   auto_ptr<EcalRecHitCollection> eerhitscleaned(new EcalRecHitCollection()); 
   for(unsigned int i = 0;  i < eerhitsuncleaned->size(); i++){
     const EcalRecHit & rhit = (*eerhitsuncleaned)[i];
     bool isclean(true);
     edm::RefVector<EcalRecHitCollection> refbeamhalorechits =  TheSummaryHalo.GetEERechits();
     for(unsigned int j = 0; j <refbeamhalorechits.size() ; j++){
       const EcalRecHit &rhitbeamhalo = *(refbeamhalorechits)[j];
       if( rhit.detid() == rhitbeamhalo.detid() ) { 
	 isclean  = false;
	 break;
       }
     }
     if(isclean) eerhitscleaned->push_back(rhit);
   }

   //  HBHERecHit
   auto_ptr<HBHERecHitCollection> hbherhitscleaned(new HBHERecHitCollection()); 
   for(unsigned int i = 0;  i < hbherhitsuncleaned->size(); i++){
     const HBHERecHit & rhit = (*hbherhitsuncleaned)[i];
     bool isclean(true);
     edm::RefVector<HBHERecHitCollection> refbeamhalorechits =  TheSummaryHalo.GetHBHERechits();
     for(unsigned int j = 0; j <refbeamhalorechits.size() ; j++){
       const HBHERecHit &rhitbeamhalo = *(refbeamhalorechits)[j];
       if( rhit.detid() == rhitbeamhalo.detid() ) { 
	 isclean  = false;
	 break;
       }
     }  
     if(isclean) hbherhitscleaned->push_back(rhit);
   }



   iEvent.put(ebrhitscleaned,"EcalRecHitsEB");
   iEvent.put(eerhitscleaned,"EcalRecHitsEE");
   iEvent.put(hbherhitscleaned);

}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
CaloRecHitsBeamHaloCleaned::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
CaloRecHitsBeamHaloCleaned::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
CaloRecHitsBeamHaloCleaned::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
CaloRecHitsBeamHaloCleaned::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
CaloRecHitsBeamHaloCleaned::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
CaloRecHitsBeamHaloCleaned::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CaloRecHitsBeamHaloCleaned::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloRecHitsBeamHaloCleaned);
