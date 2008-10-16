
#include "HLTriggerOffline/Tau/interface/HLTTauAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"


#include "DataFormats/Math/interface/deltaR.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using std::cout;
using std::endl;
using namespace reco;
using namespace edm;
using namespace l1extra;
using namespace trigger;

HLTTauAnalyzer::HLTTauAnalyzer(const edm::ParameterSet& iConfig):

 debug              ( iConfig.getParameter<int>                 ("debug")                    ),
 nbTaus             ( iConfig.getParameter<int>                 ("nbTaus")                   ),
 nbLeps             ( iConfig.getParameter<int>                 ("nbLeps")                   ),

 mcProducts         ( iConfig.getParameter<VInputTag>           ("mcProducts")               ),
 mcDeltaRTau        ( iConfig.getParameter<double>              ("mcDeltaRTau")              ),
 mcDeltaRLep        ( iConfig.getParameter<double>              ("mcDeltaRLep")              ),

 l1TauTrigger       ( iConfig.getParameter<std::string>         ("l1TauTrigger")             ),
 metReco            ( iConfig.getParameter<edm::InputTag>       ("HLTMETFilter")             ),
 usingMET           ( iConfig.getParameter<bool>                ("UsingMET")                 ),

 l2TauJets          ( iConfig.getParameter<VInputTag>           ("l2TauJets")                ),
 l2TauJetsFiltered  ( iConfig.getParameter<VInputTag>           ("l2TauJetsFiltered")        ),
 l25TauJets         ( iConfig.getParameter<VInputTag>           ("l25TauJets")               ),
 l3TauJets          ( iConfig.getParameter<VInputTag>           ("l3TauJets")                ),

 hltLeptonSrc       ( iConfig.getParameter<VInputTag>           ("hltLeptonSrc")             ),

 rootFile           ( iConfig.getUntrackedParameter<std::string>("rootFile","HLTPaths.root") ),
 logFile            ( iConfig.getUntrackedParameter<std::string>("logFile","HLTPaths.log")   ),
 isSignal           ( iConfig.getParameter<bool>                ("isSignal")                 )

{

  nEventsTot = 0;
  nEventsTotMcMatched =  0;
  nEventsL1 = nEventsL1McMatched =0;
  nEventsL2 = nEventsL2McMatched = 0;
  nEventsL2MET = 0;
  nEventsL2Filtered = nEventsL2FilteredMcMatched = 0;
  nEventsL25 = nEventsL25McMatched = 0;
  nEventsL3 = nEventsL3McMatched = 0;
 

 // Create output ROOT file and histograms
 tauFile  = new TFile( rootFile.c_str(),"recreate");
  
  float BinsEt [3] = { 8.,0.,80. }, BinsEta[3] = { 8.,-3.,3. };
  
  hMcTauEt        = new TH1F( "hMcTauEt"    ,"Monte-Carlo #tau E_{T}; E_{T}, GeV"  ,int(BinsEt[0] ),BinsEt[1] ,BinsEt[2] );
   hMcTauEt->Sumw2();
  hMcTauEta       = new TH1F( "hMcTauEta"   ,"Monte-Carlo #tau #eta; #eta"         ,int(BinsEta[0]),BinsEta[1],BinsEta[2] );
   hMcTauEta->Sumw2();
  hMcLepEt        = new TH1F( "hMcLepEt"    ,"Monte-Carlo lepton E_{T}; E_{T}, GeV",int(BinsEt[0] ),BinsEt[1] ,BinsEt[2] );
   hMcLepEt->Sumw2();
  hMcLepEta       = new TH1F( "hMcLepEta"   ,"Monte-Carlo lepton #eta; #eta"       ,int(BinsEta[0]),BinsEta[1],BinsEta[2] );
   hMcLepEta->Sumw2();

 
}

HLTTauAnalyzer::~HLTTauAnalyzer() {}

//------------------------------------------------------------------------------------------------------------------

void HLTTauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  nEventsTot++;
  MakeGeneratorAnalysis( iEvent );
    MakeLevel1Analysis( iEvent );
    if(usingMET) MakeLevel2METAnalysis( iEvent );
    MakeLevel2Analysis( iEvent );
    MakeLevel25Analysis( iEvent );
    MakeLevel3Analysis( iEvent );
}

//------------------------------------------------------------------------------------------------------------------

void HLTTauAnalyzer::MakeGeneratorAnalysis( const edm::Event& iEvent )
{
  isMcMatched = false;
 mcTauJet.clear(); mcLepton.clear(); mcNeutrina.clear();

 edm::Handle<LVColl> McInfo;

  for( VInputTag::const_iterator t = mcProducts.begin(); t != mcProducts.end(); ++t )
    {
     if( debug >= 1 ) cout << " processing product: " << t->label() << ", " << t->instance() << endl; 
     iEvent.getByLabel(*t,McInfo);

     if( !McInfo.isValid() ) continue;

     if( debug >= 1 ) cout << " -> product size: " << McInfo->size() << endl;

     if( t->instance() == "Jets"     ) mcTauJet.assign( McInfo->begin(),McInfo->end() );
     if( t->instance() == "Leptons"  ) mcLepton.assign( McInfo->begin(),McInfo->end() );
     if( t->instance() == "Neutrina" ) mcNeutrina.assign( McInfo->begin(),McInfo->end() );
    }

  if(mcTauJet.size() >= nbTaus) {
    nEventsTotMcMatched++;
    isMcMatched = true;
  }

  for( size_t i = 0; i < mcTauJet.size(); ++i ) {
   if( debug >= 1 ) 
    cout << " mc-tau: pt,eta,phi: " << mcTauJet[i].Et() << ", " << mcTauJet[i].Eta() << ", "  << mcTauJet[i].Phi() << endl;
   hMcTauEt ->Fill( mcTauJet[i].Et()  ); hMcTauEta->Fill( mcTauJet[i].Eta() ); 
  }
  for( size_t i = 0; i < mcLepton.size(); ++i ) {
   if( debug >= 1 ) 
    cout << " mc-lep: pt,eta,phi: " << mcLepton[i].Et() << ", " << mcLepton[i].Eta() << ", " << mcLepton[i].Phi() << endl;
   hMcLepEt ->Fill( mcLepton[i].Et()  ); hMcLepEta->Fill( mcLepton[i].Eta() ); 
  }

}

//------------------------------------------------------------------------------------------------------------------

void HLTTauAnalyzer::MakeLevel1Analysis( const edm::Event& iEvent )
{
  // Clear things 
  isL1Accepted = isL1McMatched = false;
  
  level1Tau.clear(); level1Lep.clear();
 // Retrieve L1 trigger decision and particles maps in order to get taus, electrons, muons 
 // (this will disapear with 1_7_0) according to the requested L1 trigger name

edm::Handle<TriggerFilterObjectWithRefs> l1TriggeredTaus;
 if(!iEvent.getByLabel(l1TauTrigger,l1TriggeredTaus)) return;

  std::vector<L1JetParticleRef> tauCandRefVec;
  l1TriggeredTaus->getObjects(trigger::TriggerL1TauJet,tauCandRefVec);

 bool  L1TauFired = false;
 if(tauCandRefVec.size() >= nbTaus) L1TauFired = true;
  
  if( L1TauFired ) {

   isL1Accepted = true;
   
   nEventsL1++;
   L1JetParticleRef tauL1CandRef;
   size_t NbMatchedTaus = 0;
   size_t NbMatchedLeps = 0;



   for( unsigned int i=0; i <tauCandRefVec.size();i++)
     {  
       tauL1CandRef = tauCandRefVec[i];
       //Avoid taking other from taujets in combined triggers
       if(typeid(*tauL1CandRef) == typeid(L1JetParticle)){
       

	 LV tau = (*tauL1CandRef).p4(); level1Tau.push_back( tau );
	 //	 cout <<"Tau L1 pt "<<tau.pt()<<endl;
	 if( isSignal ) {
	   float dR = isDrMatched( tau,mcTauJet,mcDeltaRTau );
	   //         hL1TauDrMc->Fill( dR );
	   if( debug >= 2 ) cout << "  -> DeltaR(L1 tau cand.,mc-tau): " << dR << endl;
	   if( dR <=mcDeltaRTau) { NbMatchedTaus++; }
	 }
       }
       if(typeid(*tauL1CandRef) == typeid(L1EmParticle) ||typeid(*tauL1CandRef) == typeid(L1MuonParticle) ){


	 LV lep = (*tauL1CandRef).p4(); level1Lep.push_back( lep );

	 if( isSignal  && isMcMatched) {
	   float dR = isDrMatched( lep,mcLepton,mcDeltaRLep );
	   if( debug >= 2 ) cout << "  -> DeltaR(L1 iso. em. cand.,mc-lep): " << dR << endl;
	   if( dR <=mcDeltaRLep ) { NbMatchedLeps++; }
	 }
       }
     }

   if( debug >= 1 ) {
    cout << " -> # matched taus: "    << NbMatchedTaus << endl;
    cout << " -> # matched leptons: " << NbMatchedLeps << endl;
   }

   if( (NbMatchedTaus >= nbTaus) ) {
if(debug >=1)     cout << " --> event accepted Level 1 and matched with generated particles" << endl;
     nEventsL1McMatched++;
     isL1McMatched = true;
   }
   else { if( debug >= 1 )  cout << " --> event rejected at Level 1" << endl;}
  } // if Level fired
  else {
    if( debug >= 1 )  cout << " --> event rejected at Level 1" << endl;
  }
}

//------------------------------------------------------------------------------------------------------------------
void HLTTauAnalyzer::MakeLevel2METAnalysis( const edm::Event& iEvent )
{
  isL2METAccepted = false;
  if(!isL1Accepted) return;
  edm::Handle<TriggerFilterObjectWithRefs> recoMET;
  if(!iEvent.getByLabel( metReco,recoMET )) return;
VRcalomet metRefVec;
 recoMET->getObjects(trigger::TriggerMET,metRefVec);
  if(metRefVec.size() > 0) {
    isL2METAccepted = true;
    nEventsL2MET++;
  }
  
  if(!isL2METAccepted) return;
}
//------------------------------------------------------------------------------------------------------------------
void HLTTauAnalyzer::MakeLevel2Analysis( const edm::Event& iEvent )
{
 isL2Accepted = false;
 isL2FilterAccepted   = false;
 size_t NbL2Taus = 0, NbL2TausMatched = 0;
  if(!isL1Accepted) return;
  if(!isL2METAccepted && usingMET) return;

 


//  These are the output jet collections from RecoTauTag/HLTProducers/L2TauJetProvider.cc
//  These are: single-tau, double-tau and lepton+tau
//  They are CaloJetCollection, see: http://cmslxr.fnal.gov/lxr/search?filestring=CaloJet.h&string=
//  Information about consitutents (e.g towers), em fraction, hadronic fraction can be accessed here.
 
  for( VInputTag::const_iterator t = l2TauJets.begin(); t != l2TauJets.end(); ++t )
    {
     if( debug >= 1 ) cout << " processing product: " << t->label() << ", " << t->instance() << endl; 
     
     edm::Handle<reco::CaloJetCollection> L2TauJets;
     iEvent.getByLabel( *t,L2TauJets );

     if( !L2TauJets.isValid() ) continue;

     if( debug >= 1 ) cout << " -> product size: " << L2TauJets->size() << endl;
     
     NbL2Taus += L2TauJets->size();

     const reco::CaloJetCollection & taus = *(L2TauJets.product());
     
     for( size_t i = 0 ; i < taus.size(); ++i )
       {
	if( debug >= 1 ) {
	 cout << std::setiosflags(std::ios::left) 
	      << " L2 jet: "   << endl
	      << std::setw(20) << "  pt,eta,phi,e: " 
	                       << taus[i].pt() << ", " << taus[i].eta() << ", " << taus[i].phi() << ", " << taus[i].energy() << endl
	      << std::setw(20) << "  em frac.: "     << taus[i].emEnergyFraction()       << endl
	      << std::setw(20) << "  had. frac.: "   << taus[i].energyFractionHadronic() << endl;
	}
	
        if( isSignal && isL1McMatched ) {
         float dR = isDrMatched( taus[i].p4(),mcTauJet,mcDeltaRTau );
         if( debug >= 2 ) cout << "  -> DeltaR(L2 tau,mc-tau): " << dR << endl;
	 if( dR  < mcDeltaRTau) NbL2TausMatched++;
        }
       }
    }
    
  // Check if the number of taus among all L2 collections matches the requested one
  if( NbL2Taus        >= nbTaus ) { nEventsL2++; isL2Accepted = true;}
  if( NbL2TausMatched >= nbTaus ) { nEventsL2McMatched++; isL2McMatched = true;}
    
  size_t NbL2TausFiltered = 0, NbL2TausFilteredMatched = 0;

  if(isL2Accepted) {
    edm::Handle<reco::CaloJetCollection> L2TauJetsFiltered;
    
    for( VInputTag::const_iterator t = l2TauJetsFiltered.begin(); t != l2TauJetsFiltered.end(); ++t )
      {
	iEvent.getByLabel(*t,L2TauJetsFiltered);
	
	if( debug >= 1 ) cout << " processing product: " << t->label() << ", " << t->instance() << endl; 
	
	if( !L2TauJetsFiltered.isValid() ) continue;
	
	if( debug >= 1 ) cout << " -> product size: " << L2TauJetsFiltered->size() << endl;
	
	const reco::CaloJetCollection & taus = *(L2TauJetsFiltered.product());
     
	NbL2TausFiltered += taus.size();
     
	for( size_t i = 0 ; i < taus.size(); ++i )
	  {
	
	    if(debug >=1) cout << " L2 filtered jet: " << taus[i].pt() << ", " << taus[i].eta() << ", " << taus[i].phi() << ", " << taus[i].energy() << endl;

	    if( isSignal &&  isL2McMatched) {
	      float dR = isDrMatched( taus[i].p4(),mcTauJet,mcDeltaRTau );
	      if( debug >= 2 ) cout << "  -> DeltaR(L2 tau,mc-tau): " << dR << endl;
	      if( dR <mcDeltaRTau ) NbL2TausFilteredMatched++;
	    }
	  }
      }
    
    if( NbL2TausFiltered >= nbTaus ) { 
      if( debug >= 1 ) cout << " event accepted at level2 Filter" <<endl;
      nEventsL2Filtered++;
      isL2FilterAccepted = true;
      if( NbL2TausFilteredMatched >= nbTaus ) {
	isL2FilterMcMatched = true;
	nEventsL2FilteredMcMatched++;
      }
    }
    if(!isL2FilterAccepted) if( debug >= 1 ) cout << "event rejected at level 2 Filter" << endl; 
    if(isL2FilterAccepted && !isL2FilterMcMatched) if( debug >= 1 ) cout << "event rejected at level 2 Filter MC Matched" << endl; 
  }
  if(!isL2Accepted) if( debug >= 1 ) cout << "event rejected at level 2" << endl; 
  if(isL2Accepted && !isL2McMatched) if( debug >= 1 ) cout << "event rejected at level 2 Mc Matched" << endl; 
}

//------------------------------------------------------------------------------------------------------------------

void HLTTauAnalyzer::MakeLevel25Analysis( const edm::Event& iEvent )
{
isL25Accepted = false;
isL25McMatched = false;
size_t NbL25TauJets = 0;
size_t NbL25TauMatched =0;

if( !isL2FilterAccepted ) return;

for( VInputTag::const_iterator t = l25TauJets.begin(); t != l25TauJets.end(); ++t )
{
  if( debug >= 1 ) cout << " processing product: " << t->label() << ", " << t->instance() << endl; 
  
  edm::Handle<reco::CaloJetCollection> L25TauJets;
  iEvent.getByLabel( *t,L25TauJets );
  
  if( debug >= 1 ) cout << " -> product size: "    << L25TauJets->size() << endl;
  
  if( !L25TauJets.isValid() ) continue;
  
  const reco::CaloJetCollection & taus = *( L25TauJets.product() );
  
  if( taus.size() > 0 ) {
    
    NbL25TauJets += taus.size();
    
    for( size_t i = 0; i <taus.size(); ++i )
	{
	  if(debug >= 1) cout << " L25 filtered jet: " << taus[i].pt() << ", " << taus[i].eta() << ", " << taus[i].phi() << ", " << taus[i].energy() << endl;
	  if( isSignal && isL2FilterMcMatched) {
	    float dR = isDrMatched( taus[i].p4(),mcTauJet,mcDeltaRTau );
	    if( debug >= 2 ) cout << "  -> DeltaR(L25 tau,mc-tau): " << dR << endl;
	    if( dR <mcDeltaRTau ) NbL25TauMatched++;
	  }
	  
	}
  }
}

  
 if( NbL25TauJets >= nbTaus ) {
   nEventsL25++;
   isL25Accepted = true;
 }
 if (NbL25TauMatched >= nbTaus ){
	 nEventsL25McMatched++;
	 isL25McMatched = true;
       } 
    
     if(!isL25Accepted)  if( debug >= 1 ) cout << " event rejected at L25" << endl; 
     if(isL25Accepted && !isL25McMatched)  if( debug >= 1 ) cout << " event rejected at L25 MC Matching" << endl; 

}

//------------------------------------------------------------------------------------------------------------------

void HLTTauAnalyzer::MakeLevel3Analysis( const edm::Event& iEvent )
{
  isL3Accepted = false;
  isL3McMatched = false;

       size_t NbL3TauJets = 0;
      size_t NbL3TauMatched =0;
 if( !isL25Accepted ) return;


 // This is very similar (for now with cone isolation algorithm) to level 25 as 
 // the same objects, namely IsolatedTauTagInfo are used
 
  for( VInputTag::const_iterator t = l3TauJets.begin(); t != l3TauJets.end(); ++t )
    {
     if( debug >= 1 ) cout << " processing product: " << t->label() << ", " << t->instance() << endl; 
     
     edm::Handle<reco::CaloJetCollection> L3TauJets;
     iEvent.getByLabel( *t,L3TauJets );

     if( debug >= 1 ) cout << " -> product size: "    << L3TauJets->size() << endl;

     if( !L3TauJets.isValid() ) continue;
       const reco::CaloJetCollection & taus = *( L3TauJets.product() );

     if( taus.size() > 0 ) {
     
      NbL3TauJets += taus.size();

      for( size_t i = 0; i <taus.size(); ++i )
	{
	 if(debug >=1) cout << " L3 filtered jet: " << taus[i].pt() << ", " << taus[i].eta() << ", " << taus[i].phi() << ", " << taus[i].energy() << endl;
	  NbL3TauJets++;
	  if(isSignal && isL25McMatched) {
	   float dR = isDrMatched( taus[i].p4(),mcTauJet,mcDeltaRTau );
	   if( debug >= 2 )cout << "  -> DeltaR(L3 tau,mc-tau): " << dR << endl;
	   if( dR <mcDeltaRTau ) NbL3TauMatched++;
	 }
	 
	}
      
     }
    }
  if( NbL3TauJets >= nbTaus ) {
    nEventsL3++;
    isL3Accepted = true;
  }
  if (NbL3TauMatched >= nbTaus ){
    nEventsL3McMatched++;
    isL3McMatched = true;
  } 

  if(!isL3Accepted)  if( debug >= 1 ) cout << " event rejected at L3" << endl; 
  if(isL3Accepted && !isL3McMatched)  if( debug >= 1 ) cout << " event rejected at L3 MC Matching" << endl; 
	 
  
}

// ------------ method called once each job just before starting event loop  ------------

void HLTTauAnalyzer::beginJob(const edm::EventSetup&) {}

// ------------ method called once each job just after ending the event loop  ------------

void HLTTauAnalyzer::endJob() 
{
  // Finally close output file
  tauFile->Write();
  tauFile->Close();

  std::ofstream out( logFile.c_str(),std::ios::out );
  
  out << std::setiosflags(std::ios::left) << std::setw(40)
      << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl << endl;       

  float Eff, Efferr;
  out <<"N Events Analyzed "<<nEventsTot<<endl;
  out <<"N Events MC Matched "<<nEventsTotMcMatched<<endl;
  ComputeEfficiency(nEventsTot,nEventsTotMcMatched,Eff,Efferr);
  out << "Efficiency "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;

  out  <<"L1TauJets Events "<<nEventsL1<<endl;
  out <<"L1TauJets Mc Matched Events "<<nEventsL1McMatched<<endl;
  ComputeEfficiency(nEventsTot,nEventsL1,Eff,Efferr);
  out << "Efficiency "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;  
  ComputeEfficiency(nEventsTotMcMatched,nEventsL1McMatched,Eff,Efferr);
  out << "Efficiency MC "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  out <<endl;

  out  <<"L2TauJets Events "<<nEventsL2<<endl;
  out <<"L2TauJets Mc Matched Events "<<nEventsL2McMatched<<endl;
  ComputeEfficiency(nEventsL1,nEventsL2,Eff,Efferr);
 out << "Efficiency "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
 ComputeEfficiency(nEventsL1McMatched,nEventsL2McMatched,Eff,Efferr);
  out << "Efficiency MC "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  out <<endl;


  if(usingMET){
     ComputeEfficiency(nEventsL1,nEventsL2MET,Eff,Efferr);
     out<<"MET Efficiency "<<std::setprecision(3)<<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
     out <<endl;
  out  <<"L2TauJets Events "<<nEventsL2<<endl;
   out <<"L2TauJets Mc Matched Events "<<nEventsL2McMatched<<endl;
   ComputeEfficiency(nEventsL2MET, nEventsL2,Eff,Efferr);
  out << "Efficiency "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
 
  out  <<"L2FilterTauJets Events "<<nEventsL2Filtered<<endl;
   out <<"L2FilterTauJets Mc Matched Events "<<nEventsL2FilteredMcMatched<<endl;
   ComputeEfficiency(nEventsL2,nEventsL2Filtered,Eff,Efferr);
   out << "Efficiency "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
   ComputeEfficiency(nEventsL2McMatched,nEventsL2FilteredMcMatched,Eff,Efferr);
   out << "Efficiency MC "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
   out <<endl;
    
  }else{

  out  <<"L2FilterTauJets Events "<<nEventsL2Filtered<<endl;
  out <<"L2FilterTauJets Mc Matched Events "<<nEventsL2FilteredMcMatched<<endl;
  ComputeEfficiency(nEventsL2,nEventsL2Filtered,Eff,Efferr);
  out << "Efficiency "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  ComputeEfficiency(nEventsL2McMatched,nEventsL2FilteredMcMatched,Eff,Efferr);
  out << "Efficiency MC "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  out <<endl;
    
}
  out  <<"L25TauJets Events "<<nEventsL25<<endl;
  out <<"L25TauJets Mc Matched Events "<<nEventsL25McMatched<<endl;
  ComputeEfficiency(nEventsL2Filtered,nEventsL25,Eff,Efferr);
  out << "Efficiency "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  ComputeEfficiency(nEventsL2FilteredMcMatched,nEventsL25McMatched,Eff,Efferr);
  out << "Efficiency MC "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  out <<endl;
  
  out  <<"L3TauJets Events "<<nEventsL3<<endl;
  out <<"L3TauJets Mc Matched Events "<<nEventsL3McMatched<<endl;
  ComputeEfficiency(nEventsL25,nEventsL3,Eff,Efferr);
  out << "Efficiency "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  ComputeEfficiency(nEventsL25McMatched,nEventsL3McMatched,Eff,Efferr);
  out << "Efficiency MC "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  out <<endl;

  //HLT Efficiency
  ComputeEfficiency(nEventsL1,nEventsL3,Eff,Efferr);
  out << "HLT Efficiency "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  ComputeEfficiency(nEventsL1McMatched,nEventsL3McMatched,Eff,Efferr);
  out << "HLT Efficiency Mc Matched Events "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  out <<endl;

  //Total Efficiency
  ComputeEfficiency(nEventsTot,nEventsL3,Eff,Efferr);
  out << "Total Efficiency "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  ComputeEfficiency(nEventsTotMcMatched,nEventsL3McMatched,Eff,Efferr);
  out << "Total Efficiency Mc Matched Events "<<std::setprecision(3) <<Eff<<" +/- "<<std::setprecision(2) <<Efferr<<endl;
  out <<endl;

  out.close();
  
}

//------------------------------------------------------------------------------------------------------------------

void HLTTauAnalyzer::ComputeEfficiency( const int Den, const int Num, float& Eff, float& EffErr )
{
  Eff =0.;
 EffErr = 0.;
 if( Den == 0 ) { cout << "Error: cannot compute efficiency, denominator = 0 !" << endl; return; }
 
 Eff    = 1.*Num/Den;
 EffErr = sqrt( Eff*( 1.- Eff )/Den );
}

float HLTTauAnalyzer::isDrMatched( const LV& v, const LVColl& Coll, float dRCut )
{
 float dR = 100.;
 for( size_t i = 0; i < Coll.size(); ++i ) {
  dR = deltaR( v,Coll[i] );
  if( dR < dRCut ) { break; }
  else dR = 100.;
 }
 return dR ? dR : 100.;
}













