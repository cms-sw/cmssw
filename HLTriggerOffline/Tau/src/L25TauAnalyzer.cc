// -*- C++ -*-
//
// Package:    L25TauAnalyzer
// Class:      L25TauAnalyzer
// 
/**\class L25TauAnalyzer L25TauAnalyzer.cc HLTriggerOffline/Tau/src/L25TauAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Eduardo Luiggi
//         Created:  Fri Apr  4 16:37:44 CDT 2008
// $Id$
//
//

#include "HLTriggerOffline/Tau/interface/L25TauAnalyzer.h"
#include "Math/GenVector/VectorUtil.h"

// system include files
using namespace edm;
using namespace reco;
using namespace std;

L25TauAnalyzer::L25TauAnalyzer(const edm::ParameterSet& iConfig){
  jetTagSrc_ = iConfig.getParameter<InputTag>("JetTagProd");
  jetMCTagSrc_ = iConfig.getParameter<InputTag>("JetMCTagProd");
  caloJets_ = iConfig.getParameter<InputTag>("l2CaloJets");
  rootFile_ = iConfig.getParameter<string>("outputFileName");
  rSig_ = iConfig.getParameter<double>("SignalCone");
  rMatch_ = iConfig.getParameter<double>("MatchingCone");
  rIso_ = iConfig.getParameter<double>("IsolationCone");
  ptLeadTk_ = iConfig.getParameter<double>("MinimumTransverseMomentumLeadingTrack");
  minPtIsoRing_ = iConfig.getParameter<double>("MinimumTransverseMomentumInIsolationRing");
  nTracksInIsolationRing_ = iConfig.getParameter<int>("MaximumNumberOfTracksIsolationRing");
  mcMatch_ = iConfig.getParameter<double>("MCMatching");
  signal_ = iConfig.getParameter<bool>("Signal");

  l25file = new TFile(rootFile_.c_str(),"recreate");
  l25tree = new TTree("l25tree","Level 2.5 Tau Tree");
  
  
  // add branch for L2 with pt, eta, phi, and matched at L2...
  
  //l25tree->Branch("discriminator", &discriminator, "discriminator/F";
  l25tree->Branch("jetPt", &jetPt, "jetPt/F");
  l25tree->Branch("jetE", &jetE, "jetE/F");
  l25tree->Branch("jetEta", &jetEta, "jetEta/F");
  l25tree->Branch("jetPhi", &jetPhi, "jetPhi/F");
  l25tree->Branch("isolation", &isolated, "isolation/I");
  l25tree->Branch("leadingTrkPt", &leadSignalTrackPt, "leadingTrkPt/F");
  l25tree->Branch("leadingTrkJetDeltaR", &leadTrkJetDeltaR, "leadingTrkJetDeltaR/F");
  l25tree->Branch("l2match", &l2match, "l2match/B");//Number of L2 matched jets
  l25tree->Branch("l25match", &l25match, "l25match/B");
  l25tree->Branch("ecalIsoJets", &numCaloJets, "l2ecalIsoJets/I");//number of L2 Isolated jets
  l25tree->Branch("numL25Jets", &ecalIsoJets, "numL25Jets/I");
  l25tree->Branch("numPixTrkInJet", &numPixTrkInJet, "numPixTrkInJet/I");
  l25tree->Branch("numQPixTrkInJet", &numQPixTrkInJet, "numQPixTrkInJet/I");
  l25tree->Branch("hasLeadTrk", &hasLeadTrk, "hasLeadTrk/B");
  //matchedJetsPt = new TH1F("matchedJetsPt","matchedJetsPt",75,0,150);
}


L25TauAnalyzer::~L25TauAnalyzer(){
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void L25TauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  // ------------------------ MC product stuff -------------------------------------------------------------------------

   Handle<IsolatedTauTagInfoCollection> tauTagInfoHandle;
   iEvent.getByLabel(jetTagSrc_, tauTagInfoHandle);
/*
     Handle<JetTracksAssociationCollection> jetTracksHandle;			   
     try {									   
  	iEvent.getByLabel(jetTagSrc_, jetTracksHandle); 			   
     }  									   
     catch(...){								   
  	cout << " No JetTracksAssociation Collection Found in the Event " << endl; 
     }  									   
*/ 
   Handle<LVColl> mcInfo;							 
   if(signal_){ 								 
      iEvent.getByLabel(jetMCTagSrc_, mcInfo);				 
   } 									 
   
   Handle<CaloJetCollection> caloJetHandle;			   
   iEvent.getByLabel(caloJets_, caloJetHandle);  	   
   const CaloJetCollection & caloJets = *(caloJetHandle.product());
   numCaloJets = caloJets.size();
   l2match = 0;	
   unsigned int cjIt;
   for(cjIt = 0; cjIt != caloJets.size(); ++cjIt){
      LV lVL2Calo = caloJets[cjIt].p4();
      if(signal_ )l2match = match(lVL2Calo, *mcInfo);
      
      l25match = 0;
      if(&(*tauTagInfoHandle)){
         const IsolatedTauTagInfoCollection & tauTagInfoColl = *(tauTagInfoHandle.product());
         //IsolatedTauTagInfoCollection::const_iterator i = tauTagInfo.begin();
         ecalIsoJets = tauTagInfoColl.size();
         //for (; i != tauTagInfo.end(); ++i) {  // Loop over all the IsolatedTauTagInfoCollection
         IsolatedTauTagInfo tauTagInfo = tauTagInfoColl[cjIt];	    
         LV recoJet(tauTagInfo.jet()->px(), tauTagInfo.jet()->py(),
	            tauTagInfo.jet()->pz(),tauTagInfo.jet()->energy());  		         
         												         
         jetPt = recoJet.Pt();  									         
         jetE = recoJet.E();										         
         jetEta = recoJet.Eta();									         
         jetPhi = recoJet.Phi()*180.0/TMath::Pi();							         
         												         
         												         
         numPixTrkInJet = tauTagInfo.allTracks().size();							         
         numQPixTrkInJet = tauTagInfo.selectedTracks().size();  						         
         isolated = -1; 										         
         												         
         leadTrkJetDeltaR = -1000.;									         
         const TrackRef leadTrk = tauTagInfo.leadingSignalTrack(rMatch_, ptLeadTk_);				      	      
         if(!leadTrk){  										      	      
            cout << " No leading track found " << endl; 						         
            hasLeadTrk = 0;										         
         }else{ 											         
            hasLeadTrk = 1;										      	      
            leadSignalTrackPt = (tauTagInfo.leadingSignalTrack(rMatch_, ptLeadTk_))->pt();			      	      
         												      	      

            if( tauTagInfo.discriminator(rMatch_, rSig_, rIso_, ptLeadTk_, minPtIsoRing_, nTracksInIsolationRing_)==1)   {    
               isolated = 1;										      	      
            }												      	      
            math::XYZVector leadTkMomentum = leadTrk->momentum();					      	      
            math::XYZVector jetMomentum(tauTagInfo.jet()->px(), tauTagInfo.jet()->py(), tauTagInfo.jet()->pz());		      	      
            leadTrkJetDeltaR = ROOT::Math::VectorUtil::DeltaR(jetMomentum, leadTkMomentum);		      	      
         }												         
         if(signal_)l25match = match(recoJet, *mcInfo); 						         
         else l25match = 1;										         
         												         
         //}
      }
      l25tree->Fill();										         
   }
}

bool L25TauAnalyzer::match(const LV& recoJet, const LVColl& matchingObject){
   bool matched = 0;
   if(matchingObject.size() !=0 ){
      vector<LV>::const_iterator lvIt = matchingObject.begin();
      for(;lvIt != matchingObject.end(); ++lvIt){
         double deltaR = ROOT::Math::VectorUtil::DeltaR(recoJet, *lvIt);
	 if(deltaR < mcMatch_) matched = 1;
      }
   } 
   return matched;
}

// ------------ method called once each job just before starting event loop  ------------
void L25TauAnalyzer::beginJob(const edm::EventSetup&) {
}

// ------------ method called once each job just after ending the event loop  ------------
void L25TauAnalyzer::endJob() {
   l25file->Write();
}

