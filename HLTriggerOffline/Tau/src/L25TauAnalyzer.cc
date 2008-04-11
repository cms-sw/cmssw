#include "HLTriggerOffline/Tau/interface/L25TauAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"

#include "DataFormats/Candidate/interface/CandMatchMap.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

using std::cout;
using std::cin;
using std::endl;
using std::vector;

namespace GenPart2 {
  // For sorting
  bool 
  greaterEt( const HepMC::GenParticle& a, const HepMC::GenParticle& b ) {
    return (a.momentum().et()>b.momentum().et());
  }
}


 float  BinsEt [3]    = {100.,0.,200. };
 float  BinsEta[3]    = {100.,-3.,3. };
 float  BinsPhi[3]    = {100, -TMath::Pi(), TMath::Pi()};
 float  BinsMass[3]   = {100, 0, 200 };
 float  BinsDr[3]     = {100, 0, 0.5};
 float  BinsNtrk[3]   = {20, 0, 20};
 float  BinsIsoDisc[6]= {10, 0.2, 0.7, 30, 0, 1.5};
 float  BinsPt[3]     = {100, 0, 100 };
 float  BinsChi2[3]   = {100, 0, 20};
 float  BinsRecHits[3]= {13, 0, 13};
 float  BinsPixHits[4]= {4, 0, 4};
 float  BinsD0[3]     = {100, -0.1, 0.1};
 float  BinsZ0[3]     = {100, -10, 10};
 float  BinsLP[3]     = {500, 0, 5.0};
 float  BinsLP2[3]    = {100, -5.0, 5.0};


L25TauAnalyzer::L25TauAnalyzer(const edm::ParameterSet& iConfig):

 debug              ( iConfig.getParameter<int>                 ("debug")                    ),
 nbTaus             ( iConfig.getParameter<int>                 ("nbTaus")                   ),
 nbLeps             ( iConfig.getParameter<int>                 ("nbLeps")                   ),

 mcProducts         ( iConfig.getParameter<VInputTag>           ("mcProducts")               ),
 mcDeltaRTau        ( iConfig.getParameter<double>              ("mcDeltaRTau")              ),
 mcDeltaRLep        ( iConfig.getParameter<double>              ("mcDeltaRLep")              ),

 l1ParticleMap      ( iConfig.getParameter<edm::InputTag>       ("l1ParticleMap")            ),
 l1TauTrigger       ( iConfig.getParameter<std::string>         ("l1TauTrigger")             ),
 l2TauInfoAssoc_    ( iConfig.getParameter<edm::InputTag>       ("L2InfoAssociationInput")   ),

 l2TauJets          ( iConfig.getParameter<VInputTag>           ("l2TauJets")                ),
 l2TauJetsFiltered  ( iConfig.getParameter<VInputTag>           ("l2TauJetsFiltered")        ),
 l25TauJets         ( iConfig.getParameter<VInputTag>           ("l25TauJets")               ),
 l3TauJets          ( iConfig.getParameter<VInputTag>           ("l3TauJets")                ),
 hltLeptonSrc       ( iConfig.getParameter<VInputTag>           ("hltLeptonSrc")             ),
 _GeneratorSource   ( iConfig.getParameter<std::string>("GeneratorSource") ),
 rootFile           ( iConfig.getUntrackedParameter<std::string>("rootFile","HLTPaths.root") ),
 logFile            ( iConfig.getUntrackedParameter<std::string>("logFile","HLTPaths.log")   ),
 isSignal           ( iConfig.getParameter<bool>                ("isSignal")                 ),
 passAll            ( iConfig.getParameter<bool>                ("passAll")                  )
{

 nEventsL25 = 0;
 
 for( int i = 0; i < 10; ++i ) { nEventsL25Riso[i] = 0; }

 // Create output ROOT file and histograms
 tauFile  = new TFile( rootFile.c_str(),"recreate");
  
  
  
  hMcTauEt        = new TH1F( "hMcTauEt"    ,"Monte-Carlo #tau E_{T}; E_{T}, GeV"  ,int(BinsEt[0] ),BinsEt[1] ,BinsEt[2] );
   hMcTauEt->Sumw2();
  hMcTauEta       = new TH1F( "hMcTauEta"   ,"Monte-Carlo #tau #eta; #eta"         ,int(BinsEta[0]),BinsEta[1],BinsEta[2] );
   hMcTauEta->Sumw2();
  hMcLepEt        = new TH1F( "hMcLepEt"    ,"Monte-Carlo lepton E_{T}; E_{T}, GeV",int(BinsEt[0] ),BinsEt[1] ,BinsEt[2] );
   hMcLepEt->Sumw2();
  hMcLepEta       = new TH1F( "hMcLepEta"   ,"Monte-Carlo lepton #eta; #eta"       ,int(BinsEta[0]),BinsEta[1],BinsEta[2] );
   hMcLepEta->Sumw2();

   hL25JetEtL2Bare        = new TH1F( "hL25JetEtL2Bare",        "Level 2.5 Tau-jet E_{T}",       int(BinsEt[0]),  BinsEt[1],  BinsEt[2]);
   hL25JetEtaL2Bare       = new TH1F( "hL25JetEtaL2Bare",       "Level 2.5 Tau-jet #eta",       int(BinsEta[0]), BinsEta[1], BinsEta[2]);
   hL25JetPhiL2Bare       = new TH1F( "hL25JetPhiL2Bare",       "Level 2.5 Tau-jet #phi",       int(BinsPhi[0]), BinsPhi[1], BinsPhi[2]); 

   hL25JetEtL2BareMatched        = new TH1F( "hL25JetEtL2BareMatched",        "Level 2.5 Tau-jet E_{T}",       int(BinsEt[0]),  BinsEt[1],  BinsEt[2]);
   hL25JetEtaL2BareMatched       = new TH1F( "hL25JetEtaL2BareMatched",       "Level 2.5 Tau-jet #eta",       int(BinsEta[0]), BinsEta[1], BinsEta[2]);
   hL25JetPhiL2BareMatched       = new TH1F( "hL25JetPhiL2BareMatched",       "Level 2.5 Tau-jet #phi",       int(BinsPhi[0]), BinsPhi[1], BinsPhi[2]); 


   hMCTauTrueEt   = new TH1F( "hMCTauTrueEt",   "MC Tau True p_{T}",     int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hMCTauVisibleEt= new TH1F( "hMCTauVisibleEt","MC Tau Visible p_{T}",  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hMCTauEta      = new TH1F( "hMCTauEta",      "MC Tau #eta",     int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hMCTauPhi      = new TH1F( "hMCTauPhi",      "MC Tau #phi",     int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );
   hMCTauNProngs  = new TH1F( "hMCTauNProngs",  "Number of MC Prongs",   5, 0, 5 );
   hMCTrkPt       = new TH1F( "hMCTrkPt",       "MC Track P_{T}",        int(BinsPt[0]), BinsPt[1], BinsPt[2] );
   hMCTrkEta      = new TH2F( "hMCTrkEta",      "MC Track #eta Versus Track p_{T} Cut", 7, 0, 7,   int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hMCTrkPhi      = new TH2F( "hMCTrkPhi",      "MC Track #phi Versus Track p_{T} Cut", 7, 0, 7,   int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );
   hMCNTrk        = new TH2F( "hMCNTrk",        "Number of MC Tracks", 7, 0, 7, int(BinsNtrk[0]), BinsNtrk[1], BinsNtrk[2] );

   hL25Acc          = new TH1F( "hL25Acc",          "Level 2.5 Accept Rate",                2, 0, 2);
   hL25Matched      = new TH1F( "hL25Matched",      "Level 2.5 Accept Rate for Matched",    2, 0, 2);
   hL25JetEt        = new TH1F( "hL25JetEt",        "Level 2.5 Tau-jet E_{T}",       int(BinsEt[0]),  BinsEt[1],  BinsEt[2]);
   hL25JetEta       = new TH1F( "hL25JetEta",       "Level 2.5 Tau-jet #eta",       int(BinsEta[0]), BinsEta[1], BinsEta[2]);
   hL25JetPhi       = new TH1F( "hL25JetPhi",       "Level 2.5 Tau-jet #phi",       int(BinsPhi[0]), BinsPhi[1], BinsPhi[2]); 
   hL25JetEtMatched = new TH1F( "hL25JetEtMatched", "Level 2.5 Tau-jet E_{T}",       int(BinsEt[0]),  BinsEt[1],  BinsEt[2]);
   hL25JetEtaMatched= new TH1F( "hL25JetEtaMatched","Level 2.5 Tau-jet #eta",       int(BinsEta[0]), BinsEta[1], BinsEta[2]);
   hL25JetPhiMatched= new TH1F( "hL25JetPhiMatched","Level 2.5 Tau-jet #phi",       int(BinsPhi[0]), BinsPhi[1], BinsPhi[2]); 
   hL25JetDr        = new TH1F( "hL25JetDr",        "Level 2.5 Tau-jet #Delta R",       int(BinsDr[0]), BinsDr[1], BinsDr[2]); 
   hL25JetIsoDisc   = new TH2F( "hL25JetIsoDisc",   "Level 2.5 Tau-jet Isolation Discriminant", 
				int(BinsIsoDisc[0]), BinsIsoDisc[1], BinsIsoDisc[2],
				int(BinsIsoDisc[3]), BinsIsoDisc[4], BinsIsoDisc[5]);
   hL25Trk1NTrk     = new TH2F("hL25Trk1NTrk",      "Level 2.5 Number of tracks within cone of Track 1",
			       int(BinsPt[0]), BinsPt[1], BinsPt[2], int(BinsDr[0]), BinsDr[1], BinsDr[2]);
   hL25Trk2NTrk     = new TH2F("hL25Trk2NTrk",      "Level 2.5 Number of tracks within cone of Track 2",
			       int(BinsPt[0]), BinsPt[1], BinsPt[2], int(BinsDr[0]), BinsDr[1], BinsDr[2]);
   hL25Trk1NHits    = new TH1F("hL25Trk1NHits",      "Level 2.5 Number of hits within cone of Track 1", int(BinsLP[0]), BinsLP[1], BinsLP[2] );

   hL25Trk1Hits       = new TH2F("hL25Trk1Hits", "Level 2.5 Hit Map", 
				 (int)BinsLP2[0], BinsLP2[1], BinsLP2[2],
				 (int)BinsLP2[0], BinsLP2[1], BinsLP2[2] );
   hL25Trk1Layer0Hits = new TH2F("hL25Trk1Layer0Hits", "Level 2.5 Hit Map, Layer 0", 
				 (int)BinsLP2[0], BinsLP2[1], BinsLP2[2],
				 (int)BinsLP2[0], BinsLP2[1], BinsLP2[2] );
   hL25Trk1Layer1Hits = new TH2F("hL25Trk1Layer1Hits", "Level 2.5 Hit Map, Layer 1", 
				 (int)BinsLP2[0], BinsLP2[1], BinsLP2[2],
				 (int)BinsLP2[0], BinsLP2[1], BinsLP2[2] );
   hL25Trk1Layer2Hits = new TH2F("hL25Trk1Layer2Hits", "Level 2.5 Hit Map, Layer 2", 
				 (int)BinsLP2[0], BinsLP2[1], BinsLP2[2],
				 (int)BinsLP2[0], BinsLP2[1], BinsLP2[2] );
   hL25JetNtrk      = new TH1F( "hL25JetNtrk",      "Level 2.5 Tau-jet Ntrk",        int(BinsNtrk[0]), BinsNtrk[1], BinsNtrk[2]);
   hL25TrkPt        = new TH1F( "hL25TrkPt",        "Level 2.5 Tau-track P_{T}",     int(BinsPt[0]), BinsPt[1], BinsPt[2] );
   hL25TrkChi2      = new TH1F( "hL25TrkChi2",      "Level 2.5 Tau-track #chi^{2}",  int(BinsChi2[0]), BinsChi2[1], BinsChi2[2] );
   hL25TrkRecHits   = new TH1F( "hL25TrkRecHits",   "Level 2.5 Tau-track Reco Hits", int(BinsRecHits[0]), BinsRecHits[1], BinsRecHits[2] );
   hL25TrkPixHits   = new TH1F( "hL25TrkPixHits",   "Level 2.5 Tau-track Pixel Hits",int(BinsPixHits[0]), BinsPixHits[1], BinsPixHits[2] );
   hL25TrkD0        = new TH1F( "hL25TrkD0",        "Level 2.5 Tau-track d_{0}",     int(BinsD0[0]), BinsD0[1], BinsD0[2] );
   hL25TrkZ0        = new TH1F( "hL25TrkZ0",        "Level 2.5 Tau-track z_{0}",     int(BinsZ0[0]), BinsZ0[1], BinsZ0[2] );
   hL25MTauTau      = new TH1F( "hL25MTauTau",      "Level 2.5 Jet-Jet Mass",        int(BinsMass[0]),BinsMass[1],BinsMass[2]);
   hL25MTauTauAll   = new TH1F( "hL25MTauTauAll",   "Level 2.5 Jet-Jet Mass, Unmatched", int(BinsMass[0]),BinsMass[1],BinsMass[2]);



   hL25MCTauNProngsAllMC  = new TH1F( "hL25MCTauNProngsAllMC",  "Number of MC Prongs at Level 2.5",   5, 0, 5 );
   hL25MCTauTrueEtAllMC   = new TH1F( "hL25MCTauTrueEtAllMC",   "MC Tau True p_{T}",                  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauVisibleEtAllMC= new TH1F( "hL25MCTauVisibleEtAllMC","MC Tau Visible p_{T} at Level 2.5",  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauEtaAllMC      = new TH1F( "hL25MCTauEtaAllMC",      "MC Tau #eta at Level 2.5",           int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCTauPhiAllMC      = new TH1F( "hL25MCTauPhiAllMC",      "MC Tau #phi at Level 2.5",           int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );

   hL25MCTauNProngsHadTaus  = new TH1F( "hL25MCTauNProngsHadTaus",  "Number of MC Prongs at Level 2.5",   5, 0, 5 );
   hL25MCTauTrueEtHadTaus   = new TH1F( "hL25MCTauTrueEtHadTaus",   "MC Tau True p_{T}",                  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauVisibleEtHadTaus= new TH1F( "hL25MCTauVisibleEtHadTaus","MC Tau Visible p_{T} at Level 2.5",  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauEtaHadTaus      = new TH1F( "hL25MCTauEtaHadTaus",      "MC Tau #eta at Level 2.5",           int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCTauPhiHadTaus      = new TH1F( "hL25MCTauPhiHadTaus",      "MC Tau #phi at Level 2.5",           int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );

   hL25MCTauNProngs  = new TH1F( "hL25MCTauNProngs",  "Number of MC Prongs at Level 2.5",   5, 0, 5 );
   hL25MCTauTrueEt   = new TH1F( "hL25MCTauTrueEt",   "MC Tau True p_{T}",                  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauVisibleEt= new TH1F( "hL25MCTauVisibleEt","MC Tau Visible p_{T} at Level 2.5",  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauEta      = new TH1F( "hL25MCTauEta",      "MC Tau #eta at Level 2.5",           int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCTauPhi      = new TH1F( "hL25MCTauPhi",      "MC Tau #phi at Level 2.5",           int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );

   hL25MCTauNProngsMCTrkFid  = new TH1F( "hL25MCTauNProngsMCTrkFid",  "Number of MC Prongs at Level 2.5",   5, 0, 5 );
   hL25MCTauTrueEtMCTrkFid   = new TH1F( "hL25MCTauTrueEtMCTrkFid",   "MC Tau True p_{T}",                  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauVisibleEtMCTrkFid= new TH1F( "hL25MCTauVisibleEtMCTrkFid","MC Tau Visible p_{T} at Level 2.5",  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauEtaMCTrkFid      = new TH1F( "hL25MCTauEtaMCTrkFid",      "MC Tau #eta at Level 2.5",           int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCTauPhiMCTrkFid      = new TH1F( "hL25MCTauPhiMCTrkFid",      "MC Tau #phi at Level 2.5",           int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );

   hL25MCTauNProngsPFMatch  = new TH1F( "hL25MCTauNProngsPFMatch",  "Number of MC Prongs at Level 2.5",   5, 0, 5 );
   hL25MCTauTrueEtPFMatch   = new TH1F( "hL25MCTauTrueEtPFMatch",   "MC Tau True p_{T}",                  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauVisibleEtPFMatch= new TH1F( "hL25MCTauVisibleEtPFMatch","MC Tau Visible p_{T} at Level 2.5",  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauEtaPFMatch      = new TH1F( "hL25MCTauEtaPFMatch",      "MC Tau #eta at Level 2.5",           int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCTauPhiPFMatch      = new TH1F( "hL25MCTauPhiPFMatch",      "MC Tau #phi at Level 2.5",           int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );

   hL25MCTauNProngsReco  = new TH1F( "hL25MCTauNProngsReco",  "Number of MC Prongs at Level 2.5",   5, 0, 5 );
   hL25MCTauTrueEtReco   = new TH1F( "hL25MCTauTrueEtReco",   "MC Tau True p_{T}",                  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauVisibleEtReco= new TH1F( "hL25MCTauVisibleEtReco","MC Tau Visible p_{T} at Level 2.5",  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauEtaReco      = new TH1F( "hL25MCTauEtaReco",      "MC Tau #eta at Level 2.5",           int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCTauPhiReco      = new TH1F( "hL25MCTauPhiReco",      "MC Tau #phi at Level 2.5",           int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );

   hL25MCTauNProngsSeed  = new TH1F( "hL25MCTauNProngsSeed",  "Number of MC Prongs at Level 2.5",   5, 0, 5 );
   hL25MCTauTrueEtSeed   = new TH1F( "hL25MCTauTrueEtSeed",   "MC Tau True p_{T}",                  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauVisibleEtSeed= new TH1F( "hL25MCTauVisibleEtSeed","MC Tau Visible p_{T} at Level 2.5",  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauEtaSeed      = new TH1F( "hL25MCTauEtaSeed",      "MC Tau #eta at Level 2.5",           int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCTauPhiSeed      = new TH1F( "hL25MCTauPhiSeed",      "MC Tau #phi at Level 2.5",           int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );

   hL25MCTauNProngsTrkPt  = new TH1F( "hL25MCTauNProngsTrkPt",  "Number of MC Prongs at Level 2.5",   5, 0, 5 );
   hL25MCTauTrueEtTrkPt   = new TH1F( "hL25MCTauTrueEtTrkPt",   "MC Tau True p_{T}",                  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauVisibleEtTrkPt= new TH1F( "hL25MCTauVisibleEtTrkPt","MC Tau Visible p_{T} at Level 2.5",  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauEtaTrkPt      = new TH1F( "hL25MCTauEtaTrkPt",      "MC Tau #eta at Level 2.5",           int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCTauPhiTrkPt      = new TH1F( "hL25MCTauPhiTrkPt",      "MC Tau #phi at Level 2.5",           int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );

   hL25MCTauNProngsTrkIso  = new TH1F( "hL25MCTauNProngsTrkIso",  "Number of MC Prongs at Level 2.5",   5, 0, 5 );
   hL25MCTauTrueEtTrkIso   = new TH1F( "hL25MCTauTrueEtTrkIso",   "MC Tau True p_{T}",                  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauVisibleEtTrkIso= new TH1F( "hL25MCTauVisibleEtTrkIso","MC Tau Visible p_{T} at Level 2.5",  int(BinsEt[0]), BinsEt[1], BinsEt[2] );
   hL25MCTauEtaTrkIso      = new TH1F( "hL25MCTauEtaTrkIso",      "MC Tau #eta at Level 2.5",           int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCTauPhiTrkIso      = new TH1F( "hL25MCTauPhiTrkIso",      "MC Tau #phi at Level 2.5",           int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );



   hL25MCTrkPt       = new TH1F( "hL25MCTrkPt",       "MC Track p_{T} at Level 2.5",        int(BinsPt[0]), BinsPt[1], BinsPt[2] );
   hL25MCTrkEta      = new TH2F( "hL25MCTrkEta",      "MC Track #eta Versus Track p_{T} Cut at Level 2.5", 7, 0, 7,   int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCTrkPhi      = new TH2F( "hL25MCTrkPhi",      "MC Track #phi Versus Track p_{T} Cut at Level 2.5", 7, 0, 7,   int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );   
   hL25MCNTrk        = new TH2F( "hL25MCNTrk",        "Number of MC Tracks At Level 2.5", 7, 0, 7, int(BinsNtrk[0]), BinsNtrk[1], BinsNtrk[2] );
   hL25MCTrkPtWithMatch = 
     new TH1F( "hL25MCTrkPtWithMatch", "MC Track p_{T} at Level 2.5 With Matched Track",        
	       int(BinsPt[0]), BinsPt[1], BinsPt[2] );
   hL25MCTrkEtaWithMatch= 
     new TH2F( "hL25MCTrkEtaWithMatch","MC Track #eta Versus Track p_{T} Cut at Level 2.5 With Matched Track", 7, 0, 7,   
	       int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCTrkPhiWithMatch= 
     new TH2F( "hL25MCTrkPhiWithMatch","MC Track #phi Versus Track p_{T} Cut at Level 2.5 With Matched Track", 7, 0, 7,   
	       int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );   


   hL25MCMatchedTrkPt       = new TH1F( "hL25MCMatchedTrkPt",       "Matched Track p_{T} at Level 2.5",                 int(BinsPt[0]), BinsPt[1], BinsPt[2] );
   hL25MCMatchedTrkMCPt     = new TH1F( "hL25MCMatchedTrkMCPt",     "MC p_{T} for Matched Decay Products at Level 2.5", int(BinsPt[0]), BinsPt[1], BinsPt[2] );
   hL25MCMatchedTrkDr       = new TH1F( "hL25MCMatchedTrkDr",       "Matched Track #Delta R at Level 2.5",              int(BinsDr[0]), BinsDr[1], BinsDr[2] );
   hL25MCMatchedTrkEta      = new TH2F( "hL25MCMatchedTrkEta",      "Matched Track #eta Versus Track p_{T} Cut at Level 2.5", 7, 0, 7,   int(BinsEta[0]), BinsEta[1], BinsEta[2] );
   hL25MCMatchedTrkPhi      = new TH2F( "hL25MCMatchedTrkPhi",      "Matched Track #phi Versus Track p_{T} Cut at Level 2.5", 7, 0, 7,   int(BinsPhi[0]), BinsPhi[1], BinsPhi[2] );
   hL25MCNMatchedTrk        = new TH2F( "hL25MCNMatchedTrk",        "Number of Matched Tracks", 7, 0, 7, 5, 0, 5 );
   hL25MCMatchedTrkPtVsMCTrkPt = new TH2F( "hL25MCMatchedTrkPtVsMCTrkPt", "p_{T} Of MC Track versus p_{T} Of Matched Reco Track", 
					   int(BinsPt[0]),BinsPt[1],BinsPt[2],int(BinsPt[0]),BinsPt[1],BinsPt[2] );
}

L25TauAnalyzer::~L25TauAnalyzer() {}

//------------------------------------------------------------------------------------------------------------------

void L25TauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if ( debug >= 1 ) cout << "Running analyze" << endl;
  MakeGeneratorAnalysis( iEvent, iSetup );
  if ( debug >= 1 ) cout << "Done with generator analysis" << endl;
  MakeLevel25Analysis( iEvent, iSetup );
  if ( debug >= 1 ) cout << "Done with L25 analysis" << endl;
}

//------------------------------------------------------------------------------------------------------------------

void L25TauAnalyzer::MakeGeneratorAnalysis( const edm::Event& iEvent, const edm::EventSetup & iSetup )
{

  if ( !isSignal ) return;

 mcTauJet.clear(); mcLepton.clear(); mcNeutrina.clear();

 edm::Handle<LVColl> McInfo;

  for( VInputTag::const_iterator t = mcProducts.begin(); t != mcProducts.end(); ++t )
    {
     if( debug >= 1 ) cout << " processing product: " << t->label() << ", " << t->instance() << endl; 
     iEvent.getByLabel(*t,McInfo);

     if( !McInfo.isValid() ) continue;

     if( debug >= 1 ) cout << " -> product size: " << McInfo->size() << endl;

     if( t->instance() == "Jets"     ) mcTauJet  .assign( McInfo->begin(),McInfo->end() );
     if( t->instance() == "Leptons"  ) mcLepton  .assign( McInfo->begin(),McInfo->end() );
     if( t->instance() == "Neutrina" ) mcNeutrina.assign( McInfo->begin(),McInfo->end() );
    }

//   if( mcTauJet.size() != nbTaus ) cout << " error: # nbTaus != # mc taus !!" << endl;


  for( size_t i = 0; i < mcTauJet.size(); ++i ) {
   if( debug >= 2 ) 
    cout << " mc-tau: pt,eta,phi: " << mcTauJet[i].Et() << ", " << mcTauJet[i].Eta() << ", "  << mcTauJet[i].Phi() << endl;
   hMcTauEt ->Fill( mcTauJet[i].Et()  ); hMcTauEta->Fill( mcTauJet[i].Eta() ); 
  }
  for( size_t i = 0; i < mcLepton.size(); ++i ) {
   if( debug >= 2 ) 
    cout << " mc-lep: pt,eta,phi: " << mcLepton[i].Et() << ", " << mcLepton[i].Eta() << ", " << mcLepton[i].Phi() << endl;
   hMcLepEt ->Fill( mcLepton[i].Et()  ); hMcLepEta->Fill( mcLepton[i].Eta() ); 
  }

  getGenObjects( iEvent, iSetup );


  // Fill some histograms about MC candidates before selection
  for ( std::vector<MCTauCand>::const_iterator i = _GenTaus.begin(); i != _GenTaus.end(); i++) {

    hMCTauTrueEt->Fill( i->momentum().perp() );
    hMCTauVisibleEt->Fill( i->getVisibleP4().perp() );
    hMCTauNProngs->Fill( i->getnProng() );
    hMCTauEta->Fill( i->momentum().eta() );
    hMCTauPhi->Fill( i->momentum().phi() );

    int ntrk[7] = {0};

    // Fill some histograms for visible hadronic tau decay products
    for ( std::vector<HepMC::GenParticle>::const_iterator j = i->getStableHadronicDaughters().begin();
	  j != i->getStableHadronicDaughters().end(); j++ ) {
      hMCTrkPt->Fill( j->momentum().perp() );

      // Make eta, phi, etc, plots versus the pt cut to save some processing
      int iptcut = 0;
      for ( double ptcut = 0.0; ptcut < 7.0; ptcut+= 1.0, iptcut++ ) {
	if ( j->momentum().perp() > ptcut ) {
	  ntrk[iptcut]++;
	  hMCTrkEta->Fill( ptcut, j->momentum().eta() );
	  hMCTrkPhi->Fill( ptcut, j->momentum().phi() );
	}
      }
    }

    // Make eta, phi, etc, plots versus the pt cut to save some processing
    int iptcut = 0;
    for ( double ptcut = 0.0; ptcut < 7.0; ptcut+= 1.0, iptcut++ ) {
      hMCNTrk->Fill( ptcut, ntrk[iptcut] );
    }    
  }

  if ( debug >= 2 ) {
    cout << "Gen Objects:" << endl;
    cout << "Bosons : " << endl;
    for ( std::vector<HepMC::GenParticle>::const_iterator i = _GenBosons.begin(); i != _GenBosons.end(); i++) {
      cout << " PDG ID = " << i->pdg_id() << ", Et = " << i->momentum().perp() << endl;
    }
    
    cout << "Gen Elecs : " << endl;
    for ( std::vector<HepMC::GenParticle>::const_iterator i = _GenElecs.begin(); i != _GenElecs.end(); i++) {
      cout << " PDG ID = " << i->pdg_id() << ", Et = " << i->momentum().perp() << endl;
    }

    cout << "Gen Muons : " << endl;
    for ( std::vector<HepMC::GenParticle>::const_iterator i = _GenMuons.begin(); i != _GenMuons.end(); i++) {
      cout << " PDG ID = " << i->pdg_id() << ", Et = " << i->momentum().perp() << endl;
    }

    cout << "Taus : " << endl;
    for ( std::vector<MCTauCand>::const_iterator i = _GenTaus.begin(); i != _GenTaus.end(); i++) {
      cout << " PDG ID = " << i->pdg_id() << ", Visible Et = " << i->getVisibleP4().perp() << ", N Stable Hadronic daughters : " << i->getStableHadronicDaughters().size() << endl;
    }

    cout << "Tau Electrons : " << endl;
    for ( std::vector<HepMC::GenParticle>::const_iterator i = _GenTauElecs.begin(); i != _GenTauElecs.end(); i++) {
      cout << " PDG ID = " << i->pdg_id() << ", Et = " << i->momentum().perp() << endl;  
    }

    cout << "Tau Muons : " << endl;
    for ( std::vector<HepMC::GenParticle>::const_iterator i = _GenTauMuons.begin(); i != _GenTauMuons.end(); i++) {
      cout << " PDG ID = " << i->pdg_id() << ", Et = " << i->momentum().perp() << endl;  
    }

    cout << "Tau Charged Hadrons : " << endl;
    for ( std::vector<HepMC::GenParticle>::const_iterator i = _GenTauChargedHadrons.begin(); i != _GenTauChargedHadrons.end(); i++) {
      cout << " PDG ID = " << i->pdg_id() << ", Et = " << i->momentum().perp() << endl;  
    }

  }

}

//------------------------------------------------------------------------------------------------------------------

void L25TauAnalyzer::MakeLevel25Analysis( const edm::Event& iEvent, const edm::EventSetup & iSetup )
{

//   see: http://cmslxr.fnal.gov/lxr/source/DataFormats/BTauReco/interface/IsolatedTauTagInfo.h
//        http://cmslxr.fnal.gov/lxr/source/DataFormats/BTauReco/interface/JTATagInfo.h

// Here we look at IsolatedTauTagInfo objects which are: a jet + tracks + methods to perform cone-isolation based tau-tagging

  if ( debug >= 1 ) cout << "Performing Level 2.5 Analysis" << endl;

  // Calculate efficiency for finding tracks from taus

  // Get the matching map between tracks and MC particles
  if ( debug >= 5 ) cout << "Getting tracks to gen particle map" << endl;
  edm::Handle<reco::CandMatchMap> matchMap;
  iEvent.getByLabel( "allTracksGenParticlesMatch", matchMap );

  if ( debug >= 5 ) cout << "Getting pixel rec hits collection" << endl;
  edm::Handle<SiPixelRecHitCollection> recHitColl;
  iEvent.getByLabel( "siPixelRecHits", recHitColl);

  if ( debug >= 5 ) cout << "Getting tracker geometry" << endl;
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
  const TrackerGeometry* tracker = &(* pDD);

  if ( debug >= 5 ) cout << "Getting pf tau handle" << endl;
  edm::Handle<PFTauCollection> thePFTauHandle; 
  if ( isSignal ) 
    iEvent.getByLabel("pfRecoTauProducer",thePFTauHandle); 

  if ( debug >= 5 ) cout << "Getting pf isolation" << endl;
  edm::Handle<PFTauDiscriminatorByIsolation> thePFTauDiscriminatorByIsolation; 
  if ( isSignal )
    iEvent.getByLabel("pfRecoTauDiscriminationByIsolation",thePFTauDiscriminatorByIsolation); 

  if ( debug >= 5 ) cout << "Getting L2 information" << endl;
  edm::Handle<L2TauInfoAssociation> l2TauInfoAssoc; //Handle to the input (L2 Tau Info Association)

  try {
    iEvent.getByLabel( l2TauInfoAssoc_,l2TauInfoAssoc);
  } catch (...) {
    cout << "No L2TauInfoAssociation found in the event" << endl;
  }

  size_t NbL25Taus = 0, NbL25TausMatched = 0;
  level25Tau.clear(); level25TauMatched.clear();

  if ( !l2TauInfoAssoc.isValid() )
    return; 

  if ( debug >= 5 ) cout << "Have valid tau info at l2" << endl;

  if(l2TauInfoAssoc.isValid() ) {
    for(L2TauInfoAssociation::const_iterator p = l2TauInfoAssoc->begin();p!=l2TauInfoAssoc->end();++p) {
      //Retrieve The L2TauIsolationInfo Class from the AssociationMap
      const L2TauIsolationInfo l2info = p->val;
	    
      //Retrieve the Jet From the AssociationMap
      const Jet& jet =*(p->key);

      hL25JetEtL2Bare ->Fill( jet.pt()  );
      hL25JetEtaL2Bare->Fill( jet.eta() );
      hL25JetPhiL2Bare->Fill( jet.phi() );

      bool mcMatched = false;

      for ( int i = 0; i < _GenTaus.size() && !mcMatched; i++ ) {
	MCTauCand & mcTau = _GenTaus[i];

	double eta_l2 = jet.eta();
	double phi_l2 = jet.phi();
	
	double eta_tau = mcTau.momentum().eta();
	double phi_tau = mcTau.momentum().phi();


	double dr_mc_tau =  deltaR<double>( eta_l2, phi_l2, eta_tau, phi_tau );
	if ( debug >= 2 )
	  cout << "dr = " << dr_mc_tau << endl;
	if ( dr_mc_tau < mcDeltaRTau )  {
	  mcMatched = true;
	  if ( debug >= 2) 
	    cout << "Matched!" << endl;
	}
	
      }

      bool pfMatched = false;


      for (PFTauCollection::size_type iPFTau=0;iPFTau<thePFTauHandle->size() && !pfMatched;iPFTau++) { 
	PFTauRef thePFTau(thePFTauHandle,iPFTau); 
	
	if ( (*thePFTauDiscriminatorByIsolation)[thePFTau] == 0 ) {
	  continue;
	}
      
	double eta_pf = (*thePFTau).eta();
	double phi_pf = (*thePFTau).phi();
      
	double eta_l2 = jet.eta();
	double phi_l2 = jet.phi();
      
	double dr_pf_l2 =  deltaR<double>( eta_pf, phi_pf, eta_l2, phi_l2 );
	if ( dr_pf_l2 < mcDeltaRTau )  {
	  pfMatched = true;
	}
      
      }

      if ( mcMatched && pfMatched ) {
	hL25JetEtL2BareMatched ->Fill( jet.pt()  );
	hL25JetEtaL2BareMatched->Fill( jet.eta() );
	hL25JetPhiL2BareMatched->Fill( jet.phi() );
      }
      
    }
  }


  // Calculate true trigger efficiency
  for ( int i = 0; i < _GenTaus.size(); i++ ) {
    
    if ( debug >= 5 ) cout << "Processing gen tau " << i << endl;
    MCTauCand & mcTau = _GenTaus[i];


    
    // These are true taus that have a fiducial track that passes the pt cut in them
    hL25MCTauTrueEtAllMC->Fill( mcTau.momentum().perp() );
    hL25MCTauVisibleEtAllMC->Fill( mcTau.getVisibleP4().perp() );
    hL25MCTauNProngsAllMC->Fill( mcTau.getnProng() );
    hL25MCTauEtaAllMC->Fill( mcTau.momentum().eta() );
    hL25MCTauPhiAllMC->Fill( mcTau.momentum().phi() );
    

    // Make sure it's a hadronic tau with leading track pt > 5
    if ( mcTau.getStableHadronicDaughters().size() == 0 ) continue;
    if ( debug >= 5 ) cout << "Has stable daughters, N = " << mcTau.getStableHadronicDaughters().size() << endl;
    
    
    // These are true taus that have a fiducial track that passes the pt cut in them
    hL25MCTauTrueEtHadTaus->Fill( mcTau.momentum().perp() );
    hL25MCTauVisibleEtHadTaus->Fill( mcTau.getVisibleP4().perp() );
    hL25MCTauNProngsHadTaus->Fill( mcTau.getnProng() );
    hL25MCTauEtaHadTaus->Fill( mcTau.momentum().eta() );
    hL25MCTauPhiHadTaus->Fill( mcTau.momentum().phi() );
    

    bool mcTrkFid = false;
    for ( int imctrk = 0; imctrk < mcTau.getStableHadronicDaughters().size(); imctrk++ ) {
      if ( mcTau.getStableHadronicDaughters().at(imctrk).momentum().perp() > 3.0 )
	mcTrkFid = true;
    }
      
    if ( mcTrkFid ) {

      if ( debug >= 5 ) cout << "Has stable daughters with pt > 3 GeV" << i << endl;

      // These are true taus that have a fiducial track that passes the pt cut in them
      hL25MCTauTrueEtMCTrkFid->Fill( mcTau.momentum().perp() );
      hL25MCTauVisibleEtMCTrkFid->Fill( mcTau.getVisibleP4().perp() );
      hL25MCTauNProngsMCTrkFid->Fill( mcTau.getnProng() );
      hL25MCTauEtaMCTrkFid->Fill( mcTau.momentum().eta() );
      hL25MCTauPhiMCTrkFid->Fill( mcTau.momentum().phi() );


      // Check to see that it has a PF match
      bool mcHasPFMatch = false;
      if ( debug >= 5 ) cout << "Processing PF taus, size = " << thePFTauHandle->size() << endl;
      for (PFTauCollection::size_type iPFTau=0;iPFTau<thePFTauHandle->size() && !mcHasPFMatch;iPFTau++) { 
	if ( debug >= 5 ) cout << "Processing iPFTau = " << iPFTau << endl;
	PFTauRef thePFTau(thePFTauHandle,iPFTau); 
	
	if ( (*thePFTauDiscriminatorByIsolation)[thePFTau] == 0 ) {
	  continue;
	}
	if ( debug >= 2 )
	  cout << "Have isolated tau" << endl;
      
	double eta_pf = (*thePFTau).eta();
	double phi_pf = (*thePFTau).phi();
      
	double eta_tau = mcTau.momentum().eta();
	double phi_tau = mcTau.momentum().phi();
      
	if ( debug >= 2 ) 
	  cout << "eta_pf = " << eta_pf << ", eta_tau = " << eta_tau << ", phi_pf = " << phi_pf << ", phi_tau = " << phi_tau << endl;
	  
	double dr_pf_tau =  deltaR<double>( eta_pf, phi_pf, eta_tau, phi_tau );
	if ( debug >= 2 )
	  cout << "dr = " << dr_pf_tau << endl;
	if ( dr_pf_tau < mcDeltaRTau )  {
	  mcHasPFMatch = true;
	  if ( debug >= 2) 
	    cout << "Matched!" << endl;
	}
      
      }
      if ( debug >= 5 ) cout << "Done with PF taus" << endl;


      if ( mcHasPFMatch ) {
	if ( debug >= 5 ) cout << "Has PF match" << i << endl;


	
	// These are true fiducial taus that have a PF match
	hL25MCTauTrueEtPFMatch->Fill( mcTau.momentum().perp() );
	hL25MCTauVisibleEtPFMatch->Fill( mcTau.getVisibleP4().perp() );
	hL25MCTauNProngsPFMatch->Fill( mcTau.getnProng() );
	hL25MCTauEtaPFMatch->Fill( mcTau.momentum().eta() );
	hL25MCTauPhiPFMatch->Fill( mcTau.momentum().phi() );


	// Now examine matching to L2 tau candidates
	bool matchL2 = false; 
	//If the Collection exists do work
	if(&(*l2TauInfoAssoc)) {
	  for(L2TauInfoAssociation::const_iterator p = l2TauInfoAssoc->begin();p!=l2TauInfoAssoc->end();++p) {
	    //Retrieve The L2TauIsolationInfo Class from the AssociationMap
	    const L2TauIsolationInfo l2info = p->val;
	    
	    //Retrieve the Jet From the AssociationMap
	    const Jet& jet =*(p->key);
	    
 
	    double eta_l2 = jet.eta();
	    double phi_l2 = jet.phi();
      
	    double eta_mc = mcTau.momentum().eta();
	    double phi_mc = mcTau.momentum().phi();
		
      
	    if ( debug >= 2 ) 
	      cout << "eta_l2 = " << eta_l2 << ", eta_mc = " << eta_mc << ", phi_l2 = " << phi_l2 << ", phi_mc = " << phi_mc << endl;
	  
	    double dr_l2_mc =  deltaR<double>( eta_l2, phi_l2, eta_mc, phi_mc );
	    if ( debug >= 2 )
	      cout << "dr = " << dr_l2_mc << endl;
	    if ( dr_l2_mc < mcDeltaRTau )  {
	      matchL2 = true;
	      if ( debug >= 2) 
		cout << "Matched!" << endl;
	    }
	  }
	}

	if ( !matchL2 ) continue;
	

	// Fill some histograms about MC candidates at Level 2.5 which pass PF and L2 selection

	// -------------------------------------------------------
	// ------------BASELINE FOR L2.5 ANALYSIS-----------------
	// -------------------------------------------------------
	// These are true fiducial taus that have a PF match and pass L2
	hL25MCTauTrueEt->Fill( mcTau.momentum().perp() );
	hL25MCTauVisibleEt->Fill( mcTau.getVisibleP4().perp() );
	hL25MCTauNProngs->Fill( mcTau.getnProng() );
	hL25MCTauEta->Fill( mcTau.momentum().eta() );
	hL25MCTauPhi->Fill( mcTau.momentum().phi() );
	  
	int nmatched[7] = {0};
	int nmctrk[7] = {0};
	  
	// Fill some histograms for visible hadronic tau decay products
	for ( std::vector<HepMC::GenParticle>::const_iterator j = mcTau.getStableHadronicDaughters().begin();
	      j != mcTau.getStableHadronicDaughters().end(); j++ ) {
	      
	  // Fill histograms for all tau decay products
	  hL25MCTrkPt->Fill( j->momentum().perp() );
	  // Make eta, phi, etc, plots versus the pt cut to save some processing
	  int iptcut = 0;
	  for ( double ptcut = 0.0; ptcut < 7.0; ptcut+= 1.0, iptcut++ ) {
	    if ( j->momentum().perp() > ptcut ) {
	      nmctrk[iptcut]++;
	      hL25MCTrkEta->Fill( ptcut, j->momentum().eta() );
	      hL25MCTrkPhi->Fill( ptcut, j->momentum().phi() );
	    }
	  }
	      
	}

	// Now look at L2.5 candidates
	for( VInputTag::const_iterator t = l25TauJets.begin(); t != l25TauJets.end(); ++t ) {
	  if( debug >= 1 ) cout << " processing product: " << t->label() << ", " << t->instance() << endl; 
     
	  edm::Handle<reco::IsolatedTauTagInfoCollection> L25TauJets;
	  iEvent.getByLabel( *t,L25TauJets );

	  if( debug >= 1 ) cout << " -> product size: "    << L25TauJets->size() << endl;

	  if( !L25TauJets.isValid() ) continue;
     
	  const reco::IsolatedTauTagInfoCollection & taus = *( L25TauJets.product() );

	  if( taus.size() > 0 ) {


	    int icandMatched = -1;
	    for ( int icand = 0; icand < taus.size(); icand++ ) {

	      double eta_cand = taus[icand].jet()->p4().eta();
	      double phi_cand = taus[icand].jet()->p4().phi();
      
	      double eta_tau = mcTau.momentum().eta();
	      double phi_tau = mcTau.momentum().phi();
      
	      if ( debug >= 2 ) 
		cout << "eta_cand = " << eta_cand << ", eta_tau = " << eta_tau << ", phi_cand = " << phi_cand << ", phi_tau = " << phi_tau << endl;
	  
	      double dr_cand_tau =  deltaR<double>( eta_cand, phi_cand, eta_tau, phi_tau );
	      if ( debug >= 2 )
		cout << "dr = " << dr_cand_tau << endl;
	      if ( dr_cand_tau < mcDeltaRTau )  {
		if ( debug >= 2) 
		  cout << "Matched!" << endl;
		icandMatched = icand; 
	      }
	    }
	  

	    if ( icandMatched < 0 ) continue;



	    if ( debug >= 5 ) cout << "Tau has L2.5 candidate" << i << endl;
	    // These are true fiducial taus that have a PF match, and has a L2.5 candidate
	    hL25MCTauTrueEtReco->Fill( mcTau.momentum().perp() );
	    hL25MCTauVisibleEtReco->Fill( mcTau.getVisibleP4().perp() );
	    hL25MCTauNProngsReco->Fill( mcTau.getnProng() );
	    hL25MCTauEtaReco->Fill( mcTau.momentum().eta() );
	    hL25MCTauPhiReco->Fill( mcTau.momentum().phi() );
	    
	    // Next check if there is a seed track

	    if ( taus[icandMatched].tracks().size() == 0 ) continue;
	    

	    if ( debug >= 5 ) cout << "Has matched L2 pixel tracks" << i << endl;
	    hL25MCTauTrueEtSeed->Fill( mcTau.momentum().perp() );
	    hL25MCTauVisibleEtSeed->Fill( mcTau.getVisibleP4().perp() );
	    hL25MCTauNProngsSeed->Fill( mcTau.getnProng() );
	    hL25MCTauEtaSeed->Fill( mcTau.momentum().eta() );
	    hL25MCTauPhiSeed->Fill( mcTau.momentum().phi() );

	    // Next check if the highest pt track has pt > 5
	    bool pass3Gev = false;
	    for ( int itrk = 0; itrk < taus[icandMatched].tracks().size(); itrk++ ) {
	      if ( taus[icandMatched].tracks().at(itrk)->pt() > 3.0 ) {
		pass3Gev = true;
	      }
	    }

	    if ( !pass3Gev ) continue;

	    if ( debug >= 5 ) cout << "Has matched L2 pixel tracks with lead track pt > 3 GeV" << i << endl;

	    // ----------------------------
	    /// Final plots
	    // ----------------------------
	    // These are true fiducial taus that have a PF match, and pass L2, and have leading track pt > 3
	    hL25MCTauTrueEtTrkPt->Fill( mcTau.momentum().perp() );
	    hL25MCTauVisibleEtTrkPt->Fill( mcTau.getVisibleP4().perp() );
	    hL25MCTauNProngsTrkPt->Fill( mcTau.getnProng() );
	    hL25MCTauEtaTrkPt->Fill( mcTau.momentum().eta() );
	    hL25MCTauPhiTrkPt->Fill( mcTau.momentum().phi() );



	    
      
	    for ( std::vector<HepMC::GenParticle>::const_iterator j = mcTau.getStableHadronicDaughters().begin();
		  j != mcTau.getStableHadronicDaughters().end(); j++ ) {
	      // Fill histograms for matched tau decay products
	      double dRtrk_out = -999;
	      int matchedTrackIndex = recoTrackDrMatch( j->momentum(), taus[icandMatched], dRtrk_out, 0.02 );

	      // does it match to a real tau track, and a particle flow tau?
	      if ( matchedTrackIndex >= 0 ) {

		// Fill histograms for all tau decay products with a matched track
		hL25MCTrkPtWithMatch->Fill( j->momentum().perp() );
		// Make eta, phi, etc, plots versus the pt cut to save some processing
		int iptcut = 0;
		for ( double ptcut = 0.0; ptcut < 7.0; ptcut+= 1.0, iptcut++ ) {
		  if ( j->momentum().perp() > ptcut ) {
		    hL25MCTrkEtaWithMatch->Fill( ptcut, j->momentum().eta() );
		    hL25MCTrkPhiWithMatch->Fill( ptcut, j->momentum().phi() );
		  }
		}

		hL25MCMatchedTrkPt->Fill( taus[icandMatched].tracks()[matchedTrackIndex]->pt() );
		hL25MCMatchedTrkPtVsMCTrkPt->Fill( j->momentum().perp(), taus[icandMatched].tracks()[matchedTrackIndex]->pt() );
		hL25MCMatchedTrkMCPt->Fill(  j->momentum().perp() );
		hL25MCMatchedTrkDr->Fill( dRtrk_out );
		// Make eta, phi, etc, plots versus the pt cut to save some processing
		iptcut = 0;
		for ( double ptcut = 0.0; ptcut < 7.0; ptcut+= 1.0, iptcut++ ) {
		  if ( taus[icandMatched].tracks()[matchedTrackIndex]->pt() > ptcut ) {
		    hL25MCMatchedTrkEta->Fill( ptcut, taus[icandMatched].tracks()[matchedTrackIndex]->eta() );
		    hL25MCMatchedTrkPhi->Fill( ptcut, taus[icandMatched].tracks()[matchedTrackIndex]->phi() );
		    nmatched[iptcut]++;
		  }
		}
	      }
	    }
	    // Make eta, phi, etc, plots versus the pt cut to save some processing
	    int iptcut = 0;
	    for ( double ptcut = 0.0; ptcut < 7.0; ptcut+= 1.0, iptcut++ ) {
	      hL25MCNMatchedTrk->Fill( ptcut, nmatched[iptcut] );
	      hL25MCNTrk->Fill( ptcut, nmctrk[iptcut] );
	    }

	  } // End if taus.size() > 0
	} // End loop over L25 jet collections
      } // end if mcHasPFMatch
    } // end if leading track has pt > 5

  } // end loop over mc tau jets

  if( debug >= 1 ) cout << "about to loop over l25 tau jets information" << endl;


  for( VInputTag::const_iterator t = l25TauJets.begin(); t != l25TauJets.end(); ++t ) {
    if( debug >= 1 ) cout << " processing product: " << t->label() << ", " << t->instance() << endl; 
     
    edm::Handle<reco::IsolatedTauTagInfoCollection> L25TauJets;
    iEvent.getByLabel( *t,L25TauJets );

    if( debug >= 1 ) cout << " -> product size: "    << L25TauJets->size() << endl;

    if( !L25TauJets.isValid() ) continue;
    
    if ( debug >= 2 ) cout << "Have valid L25 tau jets" << endl;
     
    const reco::IsolatedTauTagInfoCollection & taus = *( L25TauJets.product() );

    if( taus.size() > 0 ) {

      if ( debug >= 2 ) cout << "Have at least one tau jet" << endl;
     
      nEventsL25++;

      size_t NbTagTauJets = 0;

      for( size_t i = 0; i <taus.size(); ++i ) {
	edm::RefToBase<reco::Jet> jet = taus[i].jet();
	reco::TrackRefVector tracks   = taus[i].tracks();

	// Match PFtaus
	bool hasMatch = false;

	if ( isSignal ) {
	  if ( debug >= 2 ) cout << "Looking for PF" << endl;
	  for (PFTauCollection::size_type iPFTau=0;iPFTau<thePFTauHandle->size() && !hasMatch;iPFTau++) { 
	    PFTauRef thePFTau(thePFTauHandle,iPFTau); 
	   
	    if ( (*thePFTauDiscriminatorByIsolation)[thePFTau] == 0 ) {
	      continue;
	    }
	    if ( debug >= 2 )
	      cout << "Have isolated tau" << endl;
	   
	    double eta_pf = (*thePFTau).eta();
	    double phi_pf = (*thePFTau).phi();
	   
	    double eta_tau = jet->eta();
	    double phi_tau = jet->phi();

	    if ( debug >= 2 ) 
	      cout << "eta_pf = " << eta_pf << ", eta_tau = " << eta_tau << ", phi_pf = " << phi_pf << ", phi_tau = " << phi_tau << endl;
	  
	    double dr_pf_tau =  deltaR<double>( eta_pf, phi_pf, eta_tau, phi_tau );
	    if ( debug >= 2 )
	      cout << "dr = " << dr_pf_tau << endl;
	    if ( dr_pf_tau < 0.5 )  {
	      hasMatch = true;
	      if ( debug >= 2) 
		cout << "Matched!" << endl;
	    }
	   
	  }
	}
	else {
	  hasMatch = true;
	}

	if ( debug >= 2) cout << "Looking for L2 match" << endl;


	// Now examine matching to L2 tau candidates
	bool matchL2 = false; 
	//If the Collection exists do work
	if(l2TauInfoAssoc.isValid()) {
	  for(L2TauInfoAssociation::const_iterator p = l2TauInfoAssoc->begin();p!=l2TauInfoAssoc->end();++p) {
	    //Retrieve The L2TauIsolationInfo Class from the AssociationMap
	    const L2TauIsolationInfo l2info = p->val;
	    
	    //Retrieve the Jet From the AssociationMap
	    const Jet& l2jet =*(p->key);
	    
 
	    double eta_l2 = l2jet.eta();
	    double phi_l2 = l2jet.phi();
      
	    double eta_l25 = taus[i].jet()->eta();
	    double phi_l25 = taus[i].jet()->phi();
		
      
	    if ( debug >= 2 ) 
	      cout << "eta_l2 = " << eta_l2 << ", eta_l25 = " << eta_l25 << ", phi_l2 = " << phi_l2 << ", phi_l25 = " << phi_l25 << endl;
	  
	    double dr_l2_l25 =  deltaR<double>( eta_l2, phi_l2, eta_l25, phi_l25 );
	    if ( debug >= 2 )
	      cout << "dr = " << dr_l2_l25 << endl;
	    if ( dr_l2_l25 < mcDeltaRTau )  {
	      matchL2 = true;
	      if ( debug >= 2) 
		cout << "Matched!" << endl;
	    }
	  }
	}

	if ( !matchL2 ) continue;
	 	
	if ( debug >= 2 )
	  cout << "Have match" << endl;

// 	double m_cone = 0.1;
// 	double sig_cone = 0.1;
// 	double iso_cone = 0.5;
// 	double pt_min_lt = 6.0;
// 	double pt_min_tk = 1.0;
// 	int nTracksIsoRing = 0;

// 	// Select only IsolatedTauTagInfo objects that pass offline selection requirements
// 	double discriminator = taus[i].discriminator( m_cone, sig_cone, iso_cone, pt_min_lt, pt_min_tk, nTracksIsoRing );

// 	if ( discriminator < 1.0 ) continue; 

// 	if ( debug >= 2 ) 
// 	  cout << "Pass discriminator" << endl;

	level25Tau.push_back( taus[i].jet()->p4() );

	if( debug >= 1 ) {
	  jet->print();
	}
        
	hL25JetEt->Fill( taus[i].jet()->pt() );
	hL25JetEta->Fill( taus[i].jet()->eta() );
	hL25JetPhi->Fill( taus[i].jet()->phi() );
	hL25JetNtrk->Fill( tracks.size() );

	MCTauCand * mcTauCand = 0;
        if( isSignal ) {
	  float dR = isDrMatched( taus[i].jet()->p4(),mcTauJet,mcDeltaRTau );
	  hL25JetDr->Fill( dR );
	  mcTauCand = getMatchedTauCand( taus[i].jet()->p4(), mcDeltaRTau );
	  if( debug >= 2 ) cout << "  -> DeltaR(L25 tau,mc-tau): " << dR << endl;

	  // Does it match to a real tau, and a particle flow tau?
	  if( dR && hasMatch ){
	    NbL25TausMatched++;
	    level25TauMatched.push_back( taus[i].jet()->p4() );
	    hL25JetEtMatched->Fill( taus[i].jet()->pt() );
	    hL25JetEtaMatched->Fill( taus[i].jet()->eta() );
	    hL25JetPhiMatched->Fill( taus[i].jet()->phi() );
	  }
        }




	 
	if ( debug >= 1 ) cout <<" # of associated tracks " << tracks.size() << endl;
	for( reco::TrackRefVector::const_iterator it = tracks.begin(); it != tracks.end(); it++ ) {
	  if ( debug >= 1 ) {
	    cout << " Track pt "          << (*it)->pt()
		 << "  chi2 "             << (*it)->normalizedChi2()
		 << "  recHits "          << (*it)->recHitsSize()
		 << "  pixel valid hits " << (*it)->hitPattern().numberOfValidPixelHits()
		 << "  Ip(xy) "           << (*it)->d0()
		 << "  IP(z) "            << (*it)->dz()                                  
		 << endl;
	  }
	  
	  hL25TrkPt        ->Fill( (*it)->pt() );
	  hL25TrkChi2      ->Fill( (*it)->normalizedChi2()                      );
	  hL25TrkRecHits   ->Fill( (*it)->recHitsSize()                         );
	  hL25TrkPixHits   ->Fill( (*it)->hitPattern().numberOfValidPixelHits() );
	  hL25TrkD0        ->Fill( (*it)->d0()                                  );
	  hL25TrkZ0        ->Fill( (*it)->dz()                                  );
	}

	if ( tracks.size() > 0 && hasMatch ) {
	  reco::TrackRefVector::const_iterator trk1 = tracks.begin();
	  for ( reco::TrackRefVector::const_iterator it = tracks.begin(); it != tracks.end(); it++ ) {
	    if ( it != trk1 ) {
	      double eta1 = (*trk1)->eta();
	      double eta2 = (*it)  ->eta();
	      double phi1 = (*trk1)->phi();
	      double phi2 = (*it)  ->phi();
	      double dr_trki =  deltaR<double>( eta1, phi1, eta2, phi2 );
	      hL25Trk1NTrk->Fill( (*it)->pt(), dr_trki );
	    }
	  }
	}
	if ( tracks.size() > 1 && hasMatch ) {
	  reco::TrackRefVector::const_iterator trk2 = tracks.begin() + 1;
	  for ( reco::TrackRefVector::const_iterator it = tracks.begin(); it != tracks.end(); it++ ) {
	    if ( it != trk2 ) {
	      double eta1 = (*trk2)->eta();
	      double eta2 = (*it)  ->eta();
	      double phi1 = (*trk2)->phi();
	      double phi2 = (*it)  ->phi();
	      double dr_trki =  deltaR<double>( eta1, phi1, eta2, phi2 );
	      hL25Trk2NTrk->Fill( (*it)->pt(), dr_trki );
	    }
	  }
	}



	// This is totally horrendous brute force, and will be very slow, I'll
	// fix it up to be more elegant later
	if ( recHitColl->size() > 0 && tracks.size() > 0 && hasMatch ) {


	  // Loop over hits on tracks
	  vector<int> nHitsInCone((int)BinsLP[0]); 
	  for ( int iibin = 0; iibin < (int)BinsLP[0]; iibin++ ) nHitsInCone[i] = 0;

	  reco::TrackRefVector::const_iterator trk1 = tracks.begin();
	  for ( trackingRecHit_iterator it = (*trk1)->recHitsBegin(); it != (*trk1)->recHitsEnd(); ++it)  {
	    const TrackingRecHit &thit = **it;
	    // Is it a matched hit?
	    const SiPixelRecHit* matchedhit = dynamic_cast<const SiPixelRecHit*>(&thit);

	    DetId detId_fromTrack = matchedhit->geographicalId();
	    LocalPoint lp_trk = matchedhit->localPosition();

	    // 	    cout << "Local point for track, detId = " << detId_fromTrack.subdetId() << ", (x,y,z) = (" << lp_trk.x() << ", " << lp_trk.y() << ", " << lp_trk.z() << ")" << endl;
            
 
	    //-----Iterate over detunits
	    for (TrackerGeometry::DetContainer::const_iterator detit = pDD->dets().begin(); detit != pDD->dets().end(); detit++)  {
	      DetId detId = ((*detit)->geographicalId());

	      // Check to see if this detector ID is the same as from the track
	      if ( detId != detId_fromTrack ) continue;

	      // 	      cout << "Examining detId = " << detId.subdetId() << endl;
       
	      // Get hits in that detector ID
	      SiPixelRecHitCollection::range pixelrechitRange = (recHitColl.product())->get(detId);
	      SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.first;
	      SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorEnd = pixelrechitRange.second;
	      SiPixelRecHitCollection::const_iterator pixeliter = pixelrechitRangeIteratorBegin;
       
	      // 	      cout << "Looping over rechits for this detid" << endl;
	      //----Loop over rechits for this detId
	      for ( ; pixeliter != pixelrechitRangeIteratorEnd; ++pixeliter)  {
           
		// get the cluster position in local coordinates (cm) 
		LocalPoint lp = pixeliter->localPosition();
		LocalError le = pixeliter->localPositionError();

		
		// get the cluster position in global coordinates (cm)
		GlobalPoint gp = (*detit)->surface().toGlobal( lp );
		GlobalPoint gp_trk = (*detit)->surface().toGlobal( lp_trk );
		

		if ( debug >= 2) {
		  cout << "-----------" << endl;
		  cout << "hit local  point = (x,y,z) = (" << lp.x() << ", " << lp.y() << ", " << lp.z() << ")" << endl;
		  cout << "hit global point = (x,y,z) = (" << gp.x() << ", " << gp.y() << ", " << gp.z() << ")" << endl;		
		  cout << "trk local  point = (x,y,z) = (" << lp.x() << ", " << lp.y() << ", " << lp.z() << ")" << endl;
		  cout << "trk global point = (x,y,z) = (" << gp.x() << ", " << gp.y() << ", " << gp.z() << ")" << endl;
		  cout << "-----------" << endl;
		}

		double dx = (lp - lp_trk).x();
		double dy = (lp - lp_trk).y();
		double dz = (lp - lp_trk).z();

		double r = (lp - lp_trk).mag();
		double R = BinsLP[2] - BinsLP[1];

		hL25Trk1Hits->Fill( dx, dy );

		// 		cout << "r = " << r << ", R = " << R << endl;

		for ( int ir = 0; ir < (int)BinsLP[0]; ir++ ) {
		  double fr = R / (double)BinsLP[0] * (double)ir;
		  if ( r < fr ) {
		    nHitsInCone[ir]++;
		  }
		}

		// 		char ci;
		// 		cin >> ci;
           
	      } // for ( ; pixeliter != pixelrechitRangeIteratorEnd; ++pixeliter) 
	    } // for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) 
	  } // for ( trackingRecHit_iterator it = trk1->recHitsBegin(); it != trk1->recHitsEnd(); ++it) 
	  
	  for ( int iibin = 0; iibin < (int)BinsLP[0]; iibin++ ) {
	    // 	    cout << "iibin = " << iibin << ", nHitsInCone = " << nHitsInCone[iibin] << endl;
	    hL25Trk1NHits->SetBinContent( iibin , nHitsInCone[iibin] );
	  }

	  // 	  char ci;
	  // 	  cin >> ci;
	  
	} // if ( recHitColl.size() > 0 && tracks.size() > 0 ) {
	
       
      

        const reco::TrackRef leadTk = taus[i].leadingSignalTrack(0.1, 6.);
	
	if( !leadTk ) { cout << " Discriminator = 0, No Leading Track" << endl; }
	else {
	  
          if( taus[i].discriminator() > 0. ) { NbTagTauJets++; }
	  
	  // And now more refined :)
          int nbin = 10;
          for( int j = 0; j < nbin; ++j ) {
	    double r_iso = 0.2 + (double)j*0.05;
	    float discriminator = taus[i].discriminator(0.1,0.07,r_iso,6.,1.);
	    hL25JetIsoDisc->Fill( r_iso + 0.0001, discriminator ); // need to avoid straddling bins
	    if( debug >= 1 ) { 
	      cout << "  riso = " << r_iso << ", discriminator: " << discriminator << endl; 
	    }
	    nEventsL25Riso[j]++;	   
          }
	  if ( debug >= 1 )
	    cout << "Done with loop over bins for isolation" << endl;
	}
      } // end of loop over taus

      if ( debug >= 1 ) cout << "Done looping over taus to calculate discriminator" << endl;
      
      if( NbTagTauJets == nbTaus ) nEventsL25Tagged++;
    } // end if taus.size() > 0
    else { if( debug >= 1 ) cout << " event rejected at L25" << endl; }
     
  } // end of loop over level25TauJets

  if( NbL25Taus        >= nbTaus ) { nEventsL25++; hL25Acc->Fill( 1 ); } 
  else hL25Acc->Fill( 0 );
  if( NbL25TausMatched >= nbTaus ) { nEventsL25Matched++; hL25Matched->Fill( 1 ); } 
  else hL25Matched->Fill( 0 );


  // Sort lists
  sort ( level25Tau.begin(), level25Tau.end(), LVCollEtSorter() );
  sort ( level25TauMatched.begin(), level25TauMatched.end(), LVCollEtSorter() );


  // Get invariant masses for several objects
  if ( level25TauMatched.size() >= 2 ) {
    LV p1 = level25TauMatched[0];
    LV p2 = level25TauMatched[1];
    LV p = p1 + p2;
    double m = p.M();
    hL25MTauTau->Fill( m );
  }
  if ( level25Tau.size() >= 2 ) {
    LV p1 = level25Tau[0];
    LV p2 = level25Tau[1];
    LV p = p1 + p2;
    double m = p.M();
    hL25MTauTauAll->Fill( m );
  }


  if ( debug >= 1 ) cout << "Done with 2.5 analysis" << endl;
}

// ------------ method called once each job just before starting event loop  ------------

void L25TauAnalyzer::beginJob(const edm::EventSetup&) {}

// ------------ method called once each job just after ending the event loop  ------------

void L25TauAnalyzer::endJob() 
{
  // Finally close output file
  tauFile->Write();
  tauFile->Close();

  std::ofstream out( logFile.c_str(),std::ios::out );
  
   out << std::setiosflags(std::ios::left) << std::setw(40)
       << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl << endl;

  out.close();

}

//------------------------------------------------------------------------------------------------------------------

void L25TauAnalyzer::ComputeEfficiency( const int Num, const int Den, float & Eff, float & EffErr )
{
 Eff = EffErr = 0.;
 if( Den == 0 ) { cout << "Error: cannot compute efficiency, denominator = 0 !" << endl; return; }
 
 Eff    = 1.*Num/Den;
 EffErr = sqrt( Eff*( 1.- Eff )/Den );
}

float L25TauAnalyzer::isDrMatched( const LV& v, const LVColl& Coll, float dRCut )
{
 float dR = 999.;
 for( size_t i = 0; i < Coll.size(); ++i ) {
  dR = deltaR( v,Coll[i] );
  if( dR < dRCut ) { break; }
  else dR = 999.;
 }
 return dR ? dR : 999.;
}



void L25TauAnalyzer::getGenObjects(const edm::Event& iEvent, const edm::EventSetup & iSetup )
{
  //init
  _GenBosons.clear();
  _GenTaus.clear();
  _GenTauElecs.clear();
  _GenTauMuons.clear();
  _GenElecs.clear();
  _GenMuons.clear();
  _GenTauHadrons.clear();
  _GenTauChargedHadrons.clear();

  edm::Handle<edm::HepMCProduct> genEvent;
  iEvent.getByLabel(_GeneratorSource,genEvent);
  //genEvent->GetEvent()->print();

  for (HepMC::GenEvent::particle_const_iterator genParticle = 
	 genEvent->GetEvent()->particles_begin(); 
       genParticle != genEvent->GetEvent()->particles_end(); ++genParticle ) {

    // Find Z, W or H (SM + BSM)
    if ( (*genParticle)->pdg_id() == 22 || (*genParticle)->pdg_id() == 23 || 
	 (*genParticle)->pdg_id() == 24 || (*genParticle)->pdg_id() == -24 || // comment out for Hcharged->nutau
	 (*genParticle)->pdg_id() == 25 || (*genParticle)->pdg_id() == 32 || 
	 (*genParticle)->pdg_id() == 33 || 
	 (*genParticle)->pdg_id() == 34 || (*genParticle)->pdg_id() == -34 || 
	 (*genParticle)->pdg_id() == 35 || (*genParticle)->pdg_id() == 36 || 
	 (*genParticle)->pdg_id() == 37 || (*genParticle)->pdg_id() == -37 
	 ) {
      if ( (*genParticle)->status() == 3 ) {
	//std::cout << "***** Boson PdgId: " << (*genParticle)->pdg_id() << std::endl;
	_GenBosons.push_back(**genParticle);
	HepMC::GenVertex* genBosonDecayVertex = (*genParticle)->end_vertex();
	HepMC::GenVertex::particles_out_const_iterator genBosonDecayProduct = 
	  genBosonDecayVertex->particles_out_const_begin();
	if ( genBosonDecayVertex != NULL ){
	  for ( ;genBosonDecayProduct != genBosonDecayVertex->particles_out_const_end(); 
		++genBosonDecayProduct ) {
	    int pdg_id = std::abs((*genBosonDecayProduct)->pdg_id());

	    // Taus
	    HepMC::GenVertex* genTauDecayVertex = NULL;
	    if ( pdg_id == 15 ) {
	      //std::cout << "***** Tau Lepton Status: " << (*genBosonDecayProduct)->status() << std::endl;
	      //if ( (*genBosonDecayProduct)->status() == 3 ) {
		_GenTaus.push_back(**genBosonDecayProduct);
		genTauDecayVertex = (*genBosonDecayProduct)->end_vertex();	    
		//}
	    }
	    // Electrons
	    if ( pdg_id == 11 ) {
	      //std::cout << "***** Elec Lepton Status: " << (*genBosonDecayProduct)->status() << std::endl;
	      //if ( (*genBosonDecayProduct)->status() == 3 ) {
		_GenElecs.push_back(**genBosonDecayProduct);
		//}
	    }	    
	    // Muons
	    if ( pdg_id == 13 ) {
	      //std::cout << "***** Muon Lepton Status: " << (*genBosonDecayProduct)->status() << std::endl;
	      //if ( (*genBosonDecayProduct)->status() == 3 ) {
		_GenMuons.push_back(**genBosonDecayProduct);
		//}
	    }

	    // Search for stable generator level decay products of tau lepton 
	    std::vector<HepMC::GenParticle*> genTauDecayProducts;
	    if ( genTauDecayVertex != NULL ){
	      genTauDecayProducts = getGenStableDecayProducts(genTauDecayVertex);
	    } 

	    // Obtain decay modes of tau lepton 
	    int nTauHad = 0, nTauProng = 0, TauDecay = 0;
	    HepMC::GenParticle* TauNu = NULL;
	    for ( std::vector<HepMC::GenParticle*>::const_iterator genTauDecayProduct = genTauDecayProducts.begin(); genTauDecayProduct != genTauDecayProducts.end(); ++genTauDecayProduct ){
	      //if ((*genTauDecayProduct)->status() != 1) continue;
	      int dec_pdg_id = std::abs((*genTauDecayProduct)->pdg_id());
	      
	      
	      //if ( dec_pdg_id > 40 ) {
	      //if (_doPrintGenInfo)
	      //  std::cout << "----- Gen TauDecayPart (et,eta,phi): "<<(*genTauDecayProduct)->pdg_id()<<" - "<<(*genTauDecayProduct)->momentum().et()<<", "<<(*genTauDecayProduct)->momentum().eta() << ", "<<(*genTauDecayProduct)->momentum().phi()<<std::endl;
	      //}
	      
	      
	      // Charged hadron decays
	      if ( dec_pdg_id == 211 || dec_pdg_id == 321 ) {
		nTauHad++; nTauProng++; TauDecay = 3;
		_GenTaus.rbegin()->getStableHadronicDaughters().push_back( **genTauDecayProduct );
		_GenTauHadrons.push_back(**genTauDecayProduct);
		_GenTauChargedHadrons.push_back(**genTauDecayProduct);
		
		//if (_doPrintGenInfo)
		//  std::cout << "----- Gen TauHadCharged (pdgid - et,eta,phi): "<<(*genTauDecayProduct)->pdg_id()<<" - "<<(*genTauDecayProduct)->momentum().et()<<", "<<(*genTauDecayProduct)->momentum().eta() << ", "<<(*genTauDecayProduct)->momentum().phi()<<std::endl;
	      }
	      // pi0 decays
	      if ( dec_pdg_id == 111 ) {
		nTauHad++; 
		_GenTauHadrons.push_back(**genTauDecayProduct);
		//if (_doPrintGenInfo)
		//  std::cout << "----- Gen TauPi0 (et,eta,phi): "<<(*genTauDecayProduct)->momentum().et()<<", "<<(*genTauDecayProduct)->momentum().eta() << ", "<<(*genTauDecayProduct)->momentum().phi()<<std::endl;
	      }
	      // Electron
	      if ( dec_pdg_id == 11 ) {
		TauDecay = 1;
		// Set visible tau momentum to electron momentum
		_GenTaus.at(_GenTaus.size()-1).setVisibleP4((*genTauDecayProduct)->momentum());
		_GenTauElecs.push_back(**genTauDecayProduct);
	      }
	      // Muon
	      if ( dec_pdg_id == 13 ) {
		TauDecay = 2;
		// Set visible tau momentum to muon momentum
		_GenTaus.at(_GenTaus.size()-1).setVisibleP4((*genTauDecayProduct)->momentum());
		_GenTauMuons.push_back(**genTauDecayProduct);
	      }
	      
	      // Tau Neutrinos
	      if ( dec_pdg_id == 16 ) {
		TauNu = (*genTauDecayProduct);
	      }
	      // Elec Neutrinos
	      //if ( dec_pdg_id == 14 ) {
	      //}
	      // Muon Neutrinos
	      //if ( dec_pdg_id == 12 ) {
	      //}
	    }

	    // Set last included tau
	    if ( pdg_id == 15 ) {
	      if (_GenTaus.size()>0 && genTauDecayVertex != NULL) {
		_GenTaus.at(_GenTaus.size()-1).setDecayMode(TauDecay);
		_GenTaus.at(_GenTaus.size()-1).setnProng(nTauProng);
		
		// Set visible hadronic tau momentum
		if (TauNu != NULL && TauDecay == 3) {
		  _GenTaus.at(_GenTaus.size()-1).calcVisibleP4(TauNu->momentum());
		  /*
		  if (std::abs(_GenTaus.at(_GenTaus.size()-1).getVisibleP4().et() + TauNu->momentum().et()
			       - _GenTaus.at(_GenTaus.size()-1).momentum().et()) > 0.5 ) {
		    std::cout << "+++++ Gen Tau (et,eta,phi): "
			      <<_GenTaus.at(_GenTaus.size()-1).momentum().et()<<", "
			      <<_GenTaus.at(_GenTaus.size()-1).momentum().eta() << ", "
			      <<_GenTaus.at(_GenTaus.size()-1).momentum().phi()<<std::endl;
		    std::cout << "///// Gen TauNu (et,eta,phi): "<<TauNu->momentum().et()<<", "
			      <<TauNu->momentum().eta() << ", "<<TauNu->momentum().phi()<<std::endl;
		    std::cout << "----- Gen vis. Tau (et,eta,phi): "
			      <<_GenTaus.at(_GenTaus.size()-1).getVisibleP4().et()<<", "
			      <<_GenTaus.at(_GenTaus.size()-1).getVisibleP4().eta() << ", "
			      <<_GenTaus.at(_GenTaus.size()-1).getVisibleP4().phi()<<std::endl;
		  }
		  */
		} else if (TauNu == NULL && TauDecay == 3) {
		  //if ( TauDecay == 3 &&_doPrintGenInfo)
		  //if ( TauDecay == 3)
		  ////std::cout << "***** Error: no Tau neutrino found!" << std::endl;
		  ////_GenTaus.at(_GenTaus.size()-1).setVisibleP4(CLHEP::HepLorentzVector());		
		  //std::cout << "----- Gen Tau (et,eta,phi): "
		  //    <<_GenTaus.at(_GenTaus.size()-1).getVisibleP4().et()<<", "
		  //    <<_GenTaus.at(_GenTaus.size()-1).getVisibleP4().eta() << ", "
		  //    <<_GenTaus.at(_GenTaus.size()-1).getVisibleP4().phi()<<std::endl;
		}
	      }
	    }

	    
	  }  
	}
      }
    }
  }
  
  // Sort by Et
  std::sort(_GenTaus.begin(),_GenTaus.end(),GenPart2::greaterEt);
  std::sort(_GenElecs.begin(),_GenElecs.end(),GenPart2::greaterEt);
  std::sort(_GenMuons.begin(),_GenMuons.end(),GenPart2::greaterEt);
  std::sort(_GenTauElecs.begin(),_GenTauElecs.end(),GenPart2::greaterEt);
  std::sort(_GenTauMuons.begin(),_GenTauMuons.end(),GenPart2::greaterEt);
  std::sort(_GenTauHadrons.begin(),_GenTauHadrons.end(),GenPart2::greaterEt);
  std::sort(_GenTauChargedHadrons.begin(),_GenTauChargedHadrons.end(),GenPart2::greaterEt);


}



// Search for stable generator level decay products of tau lepton 
std::vector<HepMC::GenParticle*> L25TauAnalyzer::getGenStableDecayProducts(const HepMC::GenVertex* vertex)
{
  std::vector<HepMC::GenParticle*> decayProducts;
  for ( HepMC::GenVertex::particles_out_const_iterator particle = vertex->particles_out_const_begin(); 
	particle != vertex->particles_out_const_end(); ++particle ){
    int pdg_id = std::abs((*particle)->pdg_id());

    // check if particle is stable
    if ( pdg_id == 11 || pdg_id == 12 || pdg_id == 13 || pdg_id == 14 || pdg_id == 16 ||  
	 pdg_id == 111 || pdg_id == 211 || pdg_id == 321 || pdg_id == 311){
      // stable particle, identified by pdg code
      decayProducts.push_back((*particle));
    } else if ( (*particle)->end_vertex() != NULL ){
      // unstable particle, identified by non-zero decay vertex

      std::vector<HepMC::GenParticle*> addDecayProducts = getGenStableDecayProducts((*particle)->end_vertex());

      for ( std::vector<HepMC::GenParticle*>::const_iterator particle = addDecayProducts.begin(); particle != addDecayProducts.end(); ++particle ){
	decayProducts.push_back((*particle));
      }
    } else {
      // stable particle, not identified by pdg code
      decayProducts.push_back((*particle));
    }
  }
   
  return decayProducts;
}


int L25TauAnalyzer::recoTrackDrMatch( const CLHEP::HepLorentzVector & p4, const reco::IsolatedTauTagInfo & tau, double &dR_out, double dRcut ) const
{
  int ret = -1;
  double dR = dRcut;
  for ( int i = 0; i < tau.tracks().size(); i++ ) {

    const double PION_MASS = 0.13957018;
    double px = tau.tracks()[i]->px();
    double py = tau.tracks()[i]->py();
    double pz = tau.tracks()[i]->pz();
    double e  = sqrt( PION_MASS*PION_MASS + px*px + py*py + pz*pz );
    
    LV p4_trk( px,py,pz,e);
    double dRi = deltaR( p4, p4_trk );
    if ( dRi < dR ) {
      dR = dRi;
      dR_out = dR;
      ret = i;
    }
  }
  return ret;
}

L25TauAnalyzer::MatchElement 
L25TauAnalyzer::match(const reco::Jet& jet,const LVColl& McInfo)
{

  //Loop On the Collection and see if your tau jet is matched to one there
 //Also find the nearest Matched MC Particle to your Jet (to be complete)
 
 bool matched=false;
 double delta_min=100.;
 double mceta=0;
 double mcet=0;
 
 double matchingDR;

   matchingDR=0.15;


 if(McInfo.size()>0)
  for(std::vector<LV>::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
   {
     	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),*it);
	  if(delta<matchingDR)
	    {
	      matched=true;
	      if(delta<delta_min)
		{
		  delta_min=delta;
		  mceta=it->eta();
		  mcet=it->Et();
		}
	    }
   }

  //Create Struct and send it out!
  MatchElement match;
  match.matched=matched;
  match.deltar=delta_min;
  match.mcEta = mceta;
  match.mcEt = mcet;


 return match;
}

MCTauCand * L25TauAnalyzer::getMatchedTauCand( const LV & p4, double dRcut ) 
{
  MCTauCand * ret = 0;
  double dR = dRcut;
  for ( std::vector<MCTauCand>::iterator i = _GenTaus.begin(); i != _GenTaus.end(); i++ ) {
    double dRi = deltaR( p4, i->getVisibleP4() );
    if ( dRi < dR ) {
      dR = dRi;
      ret = &(*i);
    }
  }
  return ret;
}
