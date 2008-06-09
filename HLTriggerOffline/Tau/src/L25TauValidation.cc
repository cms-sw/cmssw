#include "HLTriggerOffline/Tau/interface/L25TauValidation.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>
#include <fstream>

L25TauValidation::L25TauValidation(const edm::ParameterSet& iConfig){
   jetTagSrc_ = iConfig.getParameter<edm::InputTag>("JetTagProd");
   jetMCTagSrc_ = iConfig.getParameter<edm::InputTag>("JetMCTagProd");
   caloJets_ = iConfig.getParameter<edm::InputTag>("l2CaloJets");
   rSig_ = iConfig.getParameter<double>("SignalCone");
   rMatch_ = iConfig.getParameter<double>("MatchingCone");
   rIso_ = iConfig.getParameter<double>("IsolationCone");
   ptLeadTk_ = iConfig.getParameter<double>("MinimumTransverseMomentumLeadingTrack");
   minPtIsoRing_ = iConfig.getParameter<double>("MinimumTransverseMomentumInIsolationRing");
   nTracksInIsolationRing_ = iConfig.getParameter<int>("MaximumNumberOfTracksIsolationRing");
   mcMatch_ = iConfig.getParameter<double>("MCMatching");
   signal_ = iConfig.getParameter<bool>("Signal");
   triggerTag_ = iConfig.getParameter<std::string>("TriggerTag");
   outFile_ = iConfig.getParameter<std::string>("OutputFileName");


   DQMStore* store = &*edm::Service<DQMStore>();
  
   if(store){		//Create the histograms
      
      store->setCurrentFolder(triggerTag_);

      jetPt = store->book1D("jetPt", "jetPt", 100, 0, 200);
      jetEt = store->book1D("jetEt", "jetEt", 100, 0, 200);
      jetEta = store->book1D("jetEta", "jetEta", 50, -2.5, 2.5);
      jetPhi = store->book1D("jetPhi", "jetPhi", 63, -3.14, 3.14);
      nL2EcalIsoJets = store->book1D("nL2EcalIsoJets", "nL2EcalIsoJets", 10, 0, 10);
      nL25Jets = store->book1D("nL25Jets", "nL25Jets", 10, 0, 10);
      nPxlTrksInL25Jet = store->book1D("nPxlTrksInL25Jet", "nPxlTrksInL25Jet", 30, 0, 30);
      nQPxlTrksInL25Jet = store->book1D("nQPxlTrksInL25Jet","nQPxlTrksInL25Jet", 15, 0, 15);
      signalLeadTrkPt = store->book1D("signalLeadTrkPt", "signalLeadTrkPt", 75, 0, 150);
      l25IsoJetPt = store->book1D("l25IsoJetPt", "l25IsoJetPt", 100, 0, 200);
      l25IsoJetEt = store->book1D("l25IsoJetEt", "l25IsoJetEt", 100, 0, 200);
      l25IsoJetEta = store->book1D("l25IsoJetEta", "l25IsoJetEta", 50, -2.5, 2.5);
      l25IsoJetPhi = store->book1D("l25IsoJetPhi", "l25IsoJetPhi", 63, -3.14, 3.14);
      l25EtaEff = store->book1D("l25EtaEff", "l25EtaEff", 50, -2.5, 2.5);
      l25EtEff = store->book1D("l25EtEff", "l25EtEff", 100, 0, 200);
      l25PtEff = store->book1D("l25PtEff", "l25PtEff", 100, 0, 200);
      l25PhiEff = store->book1D("l25PhiEff", "l25PhiEff", 63, -3.14, 3.14);
   }
}


L25TauValidation::~L25TauValidation(){
}



void 
L25TauValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
   using namespace edm;
   using namespace reco;
   
   Handle<IsolatedTauTagInfoCollection> tauTagInfoHandle;
   if(iEvent.getByLabel(jetTagSrc_, tauTagInfoHandle))
     {
   
       Handle<LVColl> mcInfo;							 
       if(signal_){ 								 
	 iEvent.getByLabel(jetMCTagSrc_, mcInfo);				 
       } 									 
   
       Handle<CaloJetCollection> caloJetHandle;			   
       iEvent.getByLabel(caloJets_, caloJetHandle);  	   
       const CaloJetCollection & caloJets = *(caloJetHandle.product());
       bool l2Match = 0;	
       unsigned int cjIt;
       for(cjIt = 0; cjIt != caloJets.size(); ++cjIt){
	 LV lVL2Calo = caloJets[cjIt].p4();
	 if(signal_ )l2Match = match(lVL2Calo, *mcInfo);
	 else l2Match = 1;
      
	 bool l25Match = 0;
	 if(&(*tauTagInfoHandle)){
	   const IsolatedTauTagInfoCollection & tauTagInfoColl = *(tauTagInfoHandle.product());
	   IsolatedTauTagInfo tauTagInfo = tauTagInfoColl[cjIt];	    
	   LV theJet(tauTagInfo.jet()->px(), tauTagInfo.jet()->py(),
		     tauTagInfo.jet()->pz(),tauTagInfo.jet()->energy());  		         
         												         
	   if(signal_)l25Match = match(theJet, *mcInfo); 						         
	   else l25Match = 1;												         
	   
	   if(l2Match&&l25Match){
	     jetPt->Fill(theJet.Pt()); 		  						         
	     jetEt->Fill(theJet.Et()); 		  							         
	     jetEta->Fill(theJet.Eta());		  						         
	     jetPhi->Fill(theJet.Phi());
	     nL2EcalIsoJets->Fill(caloJets.size());
	     nL25Jets->Fill(tauTagInfoColl.size());											         
	     nPxlTrksInL25Jet->Fill(tauTagInfo.allTracks().size());								    
	     nQPxlTrksInL25Jet->Fill(tauTagInfo.selectedTracks().size());							    
	     
	     const TrackRef leadTrk = tauTagInfo.leadingSignalTrack(rMatch_, ptLeadTk_);
	     if(!leadTrk) std::cout <<  "No leading track found " << std::endl;
	     else{
	       
               signalLeadTrkPt->Fill(leadTrk->pt());				 

               if(tauTagInfo.discriminator(rMatch_, rSig_, rIso_, ptLeadTk_, minPtIsoRing_,
	                                   nTracksInIsolationRing_)==1){
		 l25IsoJetEta->Fill(theJet.Eta());
		 l25IsoJetPt->Fill(theJet.Pt());
		 l25IsoJetPhi->Fill(theJet.Phi());
		 l25IsoJetEt->Fill(theJet.Et());
	       }
	     }
	   }
	 }
       }   	
     }											      	      
}


void L25TauValidation::beginJob(const edm::EventSetup&){

}


void L25TauValidation::endJob() {
   // Get the efficiency plots

  //   l25IsoJetEta->getTH1F()->Sumw2();
  // jetEta->getTH1F()->Sumw2();
   l25EtaEff->getTH1F()->Divide(l25IsoJetEta->getTH1F(), jetEta->getTH1F(), 1., 1., "b");
   
   // l25IsoJetPt->getTH1F()->Sumw2();
   //jetPt->getTH1F()->Sumw2();
   l25PtEff->getTH1F()->Divide(l25IsoJetPt->getTH1F(), jetPt->getTH1F(), 1., 1., "b");
  
   // l25IsoJetEt->getTH1F()->Sumw2();
   // jetEt->getTH1F()->Sumw2();
   l25EtEff->getTH1F()->Divide(l25IsoJetEt->getTH1F(), jetEt->getTH1F(), 1., 1., "b");
   
   // l25IsoJetPhi->getTH1F()->Sumw2();
   // jetPhi->getTH1F()->Sumw2();
   l25PhiEff->getTH1F()->Divide(l25IsoJetPhi->getTH1F(), jetPhi->getTH1F(), 1., 1., "b");
   
   
   //Write file
   if(outFile_.size()>0 &&(&*edm::Service<DQMStore>())) edm::Service<DQMStore>()->save (outFile_);

}

bool L25TauValidation::match(const LV& jet, const LVColl& matchingObject){
   bool matched = 0;
   if(matchingObject.size() !=0 ){
      std::vector<LV>::const_iterator lvIt = matchingObject.begin();
      for(;lvIt != matchingObject.end(); ++lvIt){
         double deltaR = ROOT::Math::VectorUtil::DeltaR(jet, *lvIt);
	 if(deltaR < mcMatch_) matched = 1;
      }
   } 
   return matched;
}

