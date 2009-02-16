// -*- C++ -*-
//
// Package:    HLTMuonDQMSource
// Class:      HLTMuonDQMSource
// 
/**\class HLTMuonDQMSource 

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Muriel VANDER DONCKT *:0
//         Created:  Wed Dec 12 09:55:42 CET 2007
// $Id: HLTMuonDQMSource.cc,v 1.9 2008/10/16 16:35:15 hdyoo Exp $
//
//



#include "DQM/HLTEvF/interface/HLTMuonDQMSource.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"

#include "TMath.h" 

using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;
//
// constructors and destructor
//
HLTMuonDQMSource::HLTMuonDQMSource( const edm::ParameterSet& ps ) :counterEvt_(0), nTrig_(0)

{
  parameters_ = ps;
  verbose_ = parameters_.getUntrackedParameter < bool > ("verbose", false);
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","HLT/HLTMonMuon");
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  coneSize_ = parameters_.getUntrackedParameter<double>("coneSize", 0.24);
  l2seedscollectionTag_ = parameters_.getUntrackedParameter<InputTag>("l2MuonSeedTag",edm::InputTag("hltL2MuonSeeds"));
  l3seedscollectionTag_ = parameters_.getUntrackedParameter<InputTag>("l3MuonSeedTag",edm::InputTag("hltL3TrajectorySeed"));
  l2collectionTag_ = parameters_.getUntrackedParameter<InputTag>("l2MuonTag",edm::InputTag("hltL2MuonCandidates"));
  l3collectionTag_ = parameters_.getUntrackedParameter<InputTag>("l3MuonTag",edm::InputTag("hltL3MuonCandidates"));
  l2isolationTag_ = parameters_.getUntrackedParameter<InputTag>("l2IsolationTag",edm::InputTag("hltL2MuonIsolations"));
  l3isolationTag_ = parameters_.getUntrackedParameter<InputTag>("l3IsolationTag",edm::InputTag("hltL3MuonIsolations"));

   dbe_ = 0 ;
   dbe_ = Service < DQMStore > ().operator->();
   dbe_->setVerbose(0);
 
   outputFile_ =
       parameters_.getUntrackedParameter < std::string > ("outputFile", "");
   if (outputFile_.size() != 0) {
     LogWarning("HLTMuonDQMSource") << "Muon HLT Monitoring histograms will be saved to " 
	       << outputFile_ << std::endl;
   }
   else {
     outputFile_ = "HLTMuonDQM.root";
   }
 
   bool disable =
     parameters_.getUntrackedParameter < bool > ("disableROOToutput", false);
   if (disable) {
     outputFile_ = "";
   }

   if (dbe_ != NULL) {
     dbe_->setCurrentFolder("HLT/HLTMonMuon");
   }

   std::vector<edm::ParameterSet> filters = parameters_.getParameter<std::vector<edm::ParameterSet> >("filters");
   for(std::vector<edm::ParameterSet>::iterator filterconf = filters.begin() ; filterconf != filters.end() ; filterconf++){
     theHLTCollectionLabels.push_back(filterconf->getParameter<std::string>("HLTCollectionLabels"));
   }

}


HLTMuonDQMSource::~HLTMuonDQMSource()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void HLTMuonDQMSource::beginJob(const EventSetup& context){

   if (dbe_) {
     dbe_->setCurrentFolder(monitorName_);
     dbe_->rmdir(monitorName_);
   }
 
 
   if (dbe_) {
     //dbe_->setCurrentFolder("monitorName_");
     if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
     LogInfo("HLTMuonDQMSource") << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;
     
     
     /// book some histograms here
     const int NBINS = 100; XMIN = 0; XMAX = 50;

     // create and cd into new folder
     char name[512], title[512];
     for ( int level = 1 ; level < 5 ; ++ level ) {
       if( level < 4 ) sprintf(name,"Level%i",level);
       else if (level == 4 ) sprintf(name,"Level%iSeed", level-2);
       dbe_->setCurrentFolder(monitorName_+name);
       if (level==1)hl1quality = dbe_->book1D("h1L1Quality","GMT quality Flag", 8, 0., 8.);
       if( level < 4 ) {
	 if( level > 1 ) {
           sprintf(name,"HLTMuonL%i_NMu",level);
           sprintf(title,"L%i number of muons",level);
           hNMu[level-1] = dbe_->book1D(name,title, 5, 0, 5);
           hNMu[level-1]->setAxisTitle("Number of muons", 1);
	 }
         sprintf(name,"HLTMuonL%i_pt",level);
         sprintf(title,"L%i Pt",level);
         hpt[level-1] = dbe_->book1D(name,title, NBINS, 0., 140);
         hpt[level-1]->setAxisTitle("Pt", 1);
         sprintf(name,"HLTMuonL%i_eta",level);
         sprintf(title,"L%i Muon #eta",level);
         heta[level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
         heta[level-1]->setAxisTitle("#eta", 1);
         sprintf(name,"HLTMuonL%i_phi",level);
         sprintf(title,"L%i Muon #phi",level);
         hphi[level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
         hphi[level-1]->setAxisTitle("#phi", 1);
         sprintf(name,"HLTMuonL%i_etaphi",level);
         sprintf(title,"L%i Muon #eta vs #phi",level);
         hetaphi[level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15,NBINS,-2.5, 2.5);
         hetaphi[level-1]->setAxisTitle("#phi", 1);
         hetaphi[level-1]->setAxisTitle("#eta", 2); 
         sprintf(name,"HLTMuonL%i_ptphi",level);
         sprintf(title,"L%i Muon pt vs #phi",level);         
         hptphi[level-1] = dbe_->book2D(name,title, NBINS, 0., 140.,NBINS,-3.15, 3.15);
         hptphi[level-1]->setAxisTitle("pt", 1);
         hptphi[level-1]->setAxisTitle("#phi", 2);
         sprintf(name,"HLTMuonL%i_pteta",level);
         sprintf(title,"L%i Muon pt vs #eta",level);         
         hpteta[level-1] = dbe_->book2D(name,title, NBINS, 0., 140.,NBINS,-2.5, 2.5);
         hpteta[level-1]->setAxisTitle("pt", 1);
         hpteta[level-1]->setAxisTitle("#eta", 2);
	 if( level > 1 ) {
           sprintf(name,"HLTMuonL%i_nhit",level);
           sprintf(title,"L%i Number of Valid Hits",level);         
           hnhit[level-1] = dbe_->book1D(name,title, NBINS, 0., 100.);
           hnhit[level-1]->setAxisTitle("Number of Valid Hits", 1);
	 }
         sprintf(name,"HLTMuonL%i_charge",level);
         sprintf(title,"L%i Muon Charge",level);         
         hcharge[level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
         hcharge[level-1]->setAxisTitle("Charge", 1);
       }
       else if( level == 4 ) {
         sprintf(name,"HLTMuonL%iSeed_NMu",level-2);
         sprintf(title,"L%iSeed number of muons",level-2);
         hNMu[level-1] = dbe_->book1D(name,title, 5, 0, 5);
         hNMu[level-1]->setAxisTitle("Number of muons", 1);
         sprintf(name,"HLTMuonL%iSeed_pt",level-2);
         sprintf(title,"L%iSeed Pt",level-2);
         hpt[level-1] = dbe_->book1D(name,title, NBINS, 0., 140);
         hpt[level-1]->setAxisTitle("Pt", 1);
         sprintf(name,"HLTMuonL%iSeed_eta",level-2);
         sprintf(title,"L%iSeed Muon #eta",level-2);
         heta[level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
         heta[level-1]->setAxisTitle("#eta", 1);
         sprintf(name,"HLTMuonL%iSeed_phi",level-2);
         sprintf(title,"L%iSeed Muon #phi",level-2);
         hphi[level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
         hphi[level-1]->setAxisTitle("#phi", 1);
         sprintf(name,"HLTMuonL%iSeed_etaphi",level-2);
         sprintf(title,"L%iSeed Muon #eta vs #phi",level-2);
         hetaphi[level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15,NBINS,-2.5, 2.5);
         hetaphi[level-1]->setAxisTitle("#phi", 1);
         hetaphi[level-1]->setAxisTitle("#eta", 2); 
         sprintf(name,"HLTMuonL%iSeed_ptphi",level-2);
         sprintf(title,"L%iSeed Muon pt vs #phi",level-2);         
         hptphi[level-1] = dbe_->book2D(name,title, NBINS, 0., 140.,NBINS,-3.15, 3.15);
         hptphi[level-1]->setAxisTitle("pt", 1);
         hptphi[level-1]->setAxisTitle("#phi", 2);
         sprintf(name,"HLTMuonL%iSeed_pteta",level-2);
         sprintf(title,"L%iSeed Muon pt vs #eta",level-2);         
         hpteta[level-1] = dbe_->book2D(name,title, NBINS, 0., 140.,NBINS,-2.5, 2.5);
         hpteta[level-1]->setAxisTitle("pt", 1);
         hpteta[level-1]->setAxisTitle("#eta", 2);
         //sprintf(name,"HLTMuonL%iSeed_nhit",level-2);
         //sprintf(title,"L%iSeed Number of Valid Hits",level-2);         
         //hnhit[level-1] = dbe_->book1D(name,title, NBINS, 0., 100.);
         //hnhit[level-1]->setAxisTitle("Number of Valid Hits", 1);
         sprintf(name,"HLTMuonL%iSeed_charge",level-2);
         sprintf(title,"L%iSeed Muon Charge",level-2);         
         hcharge[level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
         hcharge[level-1]->setAxisTitle("Charge", 1);
       }

       if (level>1&&level<4){
	 sprintf(name,"HLTMuonL%i_ptlx",level);
	 sprintf(title,"L%i Muon 90 percent efficiency Pt",level);
	 hptlx[level-2] = dbe_->book1D(name,title, NBINS, 0., 140);
	 hptlx[level-2]->setAxisTitle("90% efficiency Pt", 1);
	 sprintf(name,"HLTMuonL%i_dr",level);
	 sprintf(title,"L%i Muon radial impact vs BeamSpot",level);         
	 hdr[level-2] = dbe_->book1D(name,title, NBINS, -0.3, 0.3);
	 hdr[level-2]->setAxisTitle("R Impact (cm) vs BeamSpot", 1);
	 sprintf(name,"HLTMuonL%i_d0",level);
	 sprintf(title,"L%i Muon radial impact vs (0,0)",level);         
	 hd0[level-2] = dbe_->book1D(name,title, NBINS, -0.3, 0.3);
	 hd0[level-2]->setAxisTitle("R Impact (cm) vs 0,0", 1);
	 sprintf(name,"HLTMuonL%i_dz",level);
	 sprintf(title,"L%i Muon Z impact",level);         
	 hdz[level-2] = dbe_->book1D(name,title, NBINS, -25., 25.);
	 hdz[level-2]->setAxisTitle("Z impact (cm)", 1);
	 sprintf(name,"HLTMuonL%i_err0",level);
	 sprintf(title,"L%i Muon Error on Pt",level);         
	 herr0[level-2] = dbe_->book1D(name,title,NBINS, 0., 0.03);
	 herr0[level-2]->setAxisTitle("Error on Pt", 1);
         sprintf(name,"HLTMuonL%i_iso",level);
	 if (level==2)sprintf(title,"L%i Muon Energy in Isolation cone",level);         
	 else if (level==3)sprintf(title,"L%i Muon SumPt in Isolation cone",level);               
	 hiso[level-2]  = dbe_->book1D(name,title, NBINS, 0., 5./(level-1));
	 if ( level==2)hiso[level-2]->setAxisTitle("Calo Energy in Iso Cone (GeV)", 1);
	 else if ( level==3)hiso[level-2]->setAxisTitle("Sum Pt in Iso Cone (GeV)", 1);
	 sprintf(name,"HLTMuonL%i_DiMuMass",level);
	 sprintf(title,"L%i Opposite charge DiMuon invariant Mass",level);         
	 hdimumass[level-2]= dbe_->book1D(name,title, NBINS, 0., 150.);
	 hdimumass[level-2]->setAxisTitle("Di Muon Invariant Mass (GeV)");
	 sprintf(name,"HLTMuonL%i_drphi",level);
	 sprintf(title,"L%i #Deltar vs #phi",level);         
	 hdrphi[level-2] = dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
	 hdrphi[level-2]->setAxisTitle("#phi", 1);
	 hdrphi[level-2]->setAxisTitle("#Deltar", 2);
	 sprintf(name,"HLTMuonL%i_d0phi",level);
	 sprintf(title,"L%i #Delta0 vs #phi",level);         
	 hd0phi[level-2] = dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
	 hd0phi[level-2]->setAxisTitle("#phi", 1);
	 hd0phi[level-2]->setAxisTitle("#Delta0", 2);
	 sprintf(name,"HLTMuonL%i_dzeta",level);
	 sprintf(title,"L%i #Deltaz vs #eta",level);         
	 hdzeta[level-2] = dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
	 hdzeta[level-2]->setAxisTitle("#eta", 1);
	 hdzeta[level-2]->setAxisTitle("#Deltaz", 2);
       }
       if (level < 3 ) {
	 sprintf(name,"HLTMuonL%i_ptres",level);
	 sprintf(title,"L%iMuon1/Pt - L%iMuon1/Pt",level,level+1);         
	 hptres[level-1] = dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	 sprintf(title,"1/PtL%i - 1/PtL%i",level,level+1);         
	 hptres[level-1]->setAxisTitle(title, 1);
	 sprintf(name,"HLTMuonL%i_etares",level);
	 sprintf(title,"L%i Muon #Delta#eta (wrt L%i)",level,level+1);         
	 hetares[level-1] =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	 hetares[level-1]->setAxisTitle("#Delta#eta", 1);
	 sprintf(name,"HLTMuonL%i_phires",level);
	 sprintf(title,"L%i Muon #Delta#phi (wrt L%i)",level,level+1);         
	 hphires[level-1] =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	 hphires[level-1]->setAxisTitle("#Delta#phi", 1);
	 sprintf(name,"HLTMuonL%i_phiresphi",level);
	 sprintf(title,"L%i Muon #Delta#phi vs #phi ",level);         
	 hphiresphi[level-1] =dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
	 hphiresphi[level-1]->setAxisTitle("<#Delta#phi>", 2);
	 hphiresphi[level-1]->setAxisTitle("#phi", 1);
	 sprintf(name,"HLTMuonL%i_etareseta",level);
	 sprintf(title,"L%i Muon #Delta#eta vs #eta ",level);         
	 hetareseta[level-1] =dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
	 hetareseta[level-1]->setAxisTitle("<#Delta#eta>", 2);
	 hetareseta[level-1]->setAxisTitle("#eta", 1);
	 if (level  == 1 ){
	   sprintf(name,"HLTMuonL%i3_ptres",level);
	   sprintf(title,"L%iMuon1/Pt - L%iMuon1/Pt",level,level+2);         
	   hptres[level+1] = dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	   sprintf(title,"1/PtL%i - 1/PtL%i",level,level+2);         
	   hptres[level+1]->setAxisTitle(title, 1);
	   sprintf(name,"HLTMuonL%i3_etares",level);
	   sprintf(title,"L%i Muon #Delta#eta (wrt L%i)",level,level+2);         
	   hetares[level+1] =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	   hetares[level+1]->setAxisTitle("#Delta#eta", 1);
	   sprintf(name,"HLTMuonL%i3_phires",level);
	   sprintf(title,"L%i Muon #Delta#phi (wrt L%i)",level,level+2);         
	   hphires[level+1] =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	   hphires[level+1]->setAxisTitle("#Delta#phi", 1);
	   sprintf(name,"HLTMuonL%i3_phiresphi",level);
	   sprintf(title,"L%i Muon #Delta#phi vs #phi (wrt L3) ",level);         
	   hphiresphi[level+1] =dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
	   hphiresphi[level+1]->setAxisTitle("<#Delta#phi>", 2);
	   hphiresphi[level+1]->setAxisTitle("#phi", 1);
	   sprintf(name,"HLTMuonL%i3_etareseta",level);
	   sprintf(title,"L%i Muon #Delta#eta vs #eta (wrt L3) ",level);         
	   hetareseta[level+1] =dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
	   hetareseta[level+1]->setAxisTitle("<#Delta#eta>", 2);
	   hetareseta[level+1]->setAxisTitle("#eta", 1);
	 }
       }
     }
     dbe_->showDirStructure();
  
     // Muon det id is 2 pushed in bits 28:31
     const unsigned int detector_id = 2<<28;
     dbe_->tagContents(monitorName_, detector_id);
   } 
}

//--------------------------------------------------------
void HLTMuonDQMSource::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void HLTMuonDQMSource::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				      const EventSetup& context) {
  
}

// ----------------------------------------------------------
void HLTMuonDQMSource::analyze(const Event& iEvent, 
			 const EventSetup& iSetup )
{  
  if ( !dbe_) return;
  counterEvt_++;
  if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0) return;
  LogDebug("HLTMuonDQMSource") << " processing conterEvt_: " << counterEvt_ <<endl;

  bool trigFired = false;
  Handle<TriggerResults> trigResult;
  iEvent.getByLabel(InputTag("TriggerResults"), trigResult);
  if( !trigResult.failedToGet() ) {
    int ntrigs = trigResult->size();
    TriggerNames trigName;
    trigName.init(*trigResult);
    for( int itrig = 0; itrig != ntrigs; ++itrig) {
      for(unsigned int n=0; n < theHLTCollectionLabels.size() ; n++) { 
	//cout << "trigger = " << trigResult->name(itrig) << endl;
        if( trigResult->accept(trigName.triggerIndex(theHLTCollectionLabels[n])) ) trigFired = true;
      }
    }
  }
  if( !trigFired ) return;
  nTrig_++;

  Handle<RecoChargedCandidateCollection> l2mucands, l3mucands;
  iEvent.getByLabel (l2collectionTag_,l2mucands);
  iEvent.getByLabel (l3collectionTag_,l3mucands);
  RecoChargedCandidateCollection::const_iterator cand,cand2;

  Handle<L2MuonTrajectorySeedCollection> l2seeds; 
  iEvent.getByLabel (l2seedscollectionTag_,l2seeds);

  if (!l2seeds.failedToGet()) {
    hNMu[3]->Fill(l2seeds->size());
    L2MuonTrajectorySeedCollection::const_iterator l2seed;
    for (l2seed=l2seeds->begin() ; l2seed != l2seeds->end();++l2seed){
      PTrajectoryStateOnDet state=l2seed->startingState();
      float pt=state.parameters().momentum().perp();
      float eta=state.parameters().momentum().phi();
      float phi=state.parameters().momentum().eta();
      hcharge[3]->Fill(state.parameters().charge());
      hpt[3]->Fill(pt);
      //hnhit[3]->Fill(l2seeds->nhit());
      hphi[3]->Fill(phi);
      heta[3]->Fill(eta);
      hetaphi[3]->Fill(phi,eta);
      hptphi[3]->Fill(pt,phi);
      hpteta[3]->Fill(pt,eta);
      L1MuonParticleRef l1ref = l2seed->l1Particle();
      hcharge[0]->Fill(l1ref->charge());
      hpt[0]->Fill(l1ref->pt());
      hphi[0]->Fill(l1ref->phi());
      heta[0]->Fill(l1ref->eta());
      hetaphi[0]->Fill(l1ref->phi(),l1ref->eta());
      hptphi[0]->Fill(l1ref->pt(),l1ref->phi());
      hpteta[0]->Fill(l1ref->pt(),l1ref->eta());
      hl1quality->Fill(l1ref->gmtMuonCand().quality());
      if ( !l2mucands.failedToGet()) {
	for (cand=l2mucands->begin(); cand!=l2mucands->end(); ++cand) {
	  TrackRef tk = cand->get<TrackRef>();
	  RefToBase<TrajectorySeed> seed=tk->seedRef();
	  if ( (l2seed->startingState()).detId() == (seed->startingState()).detId() ) {
	    if(tk->pt()*l1ref->pt() != 0 )hptres[0]->Fill(1/tk->pt() - 1/l1ref->pt());
	    hetares[0]->Fill(tk->eta()-l1ref->eta());
	    hetareseta[0]->Fill(tk->eta(),tk->eta()-l1ref->eta());
	    hphires[0]->Fill(tk->phi()-l1ref->phi());
	    double dphi=tk->phi()-l1ref->phi();
	    if (dphi>TMath::TwoPi())dphi-=2*TMath::TwoPi();
	    else if (dphi<-TMath::TwoPi()) dphi+=TMath::TwoPi();
	    hphiresphi[0]->Fill(tk->phi(),dphi);
	    //find the L3 build from this L2
	    
 	    if (!l3mucands.failedToGet()) {
	      for (cand=l3mucands->begin(); cand!=l3mucands->end(); ++cand) {
	        TrackRef l3tk= cand->get<TrackRef>();
		cout << "L3 trajectory = " << l3tk->seedRef().castTo<Ref<L3MuonTrajectorySeedCollection> > ().isAvailable() << endl;
		if( l3tk->seedRef().castTo<Ref<L3MuonTrajectorySeedCollection> > ().isAvailable() ) {
	          if (l3tk->seedRef().castTo<Ref<L3MuonTrajectorySeedCollection> >()->l2Track() == tk){
		    if(l1ref->pt()*l3tk->pt() != 0 )hptres[2]->Fill(1/l1ref->pt() - 1/l3tk->pt());
		    hetares[2]->Fill(l1ref->eta()-l3tk->eta());
		    hetareseta[2]->Fill(l1ref->eta(),l1ref->eta()-l3tk->eta());
		    hphires[2]->Fill(l1ref->phi()-l3tk->phi());
		    double dphi=l1ref->phi()-l3tk->phi();
		    if (dphi>TMath::TwoPi())dphi-=2*TMath::TwoPi();
		    else if (dphi<-TMath::TwoPi()) dphi+=TMath::TwoPi();
		    hphiresphi[2]->Fill(l3tk->phi(),dphi);
		    //break; //plot only once per L2?
	          }//if
		}
	      }//for
	    }
	    break;
	  }
	}
      }
    }
  }



  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel("hltOfflineBeamSpot",recoBeamSpotHandle);
  if (!recoBeamSpotHandle.failedToGet())  beamSpot = *recoBeamSpotHandle;

  if (!l2mucands.failedToGet()) {
    LogDebug("HLTMuonDQMSource") << " filling L2 stuff " << endl;
    Handle<reco::IsoDepositMap> l2depMap;
    iEvent.getByLabel (l2isolationTag_,l2depMap);
    hNMu[1]->Fill(l2mucands->size());
    for (cand=l2mucands->begin(); cand!=l2mucands->end(); ++cand) {
      TrackRef tk = cand->get<TrackRef>();
      if (!l2depMap.failedToGet()) {
	  LogDebug("HLTMuonDQMSource") << " filling L2 Iso stuff " << endl;
	  if ( l2depMap->contains(tk.id()) ){
	    reco::IsoDepositMap::value_type calDeposit = (*l2depMap)[tk];
	    double dephlt = calDeposit.depositWithin(coneSize_);
	    hiso[0]->Fill(dephlt);
	  }
      }
    
      // eta cut
      hpt[1]->Fill(tk->pt());      
      double apar0 = fabs(tk->parameter(0));
      if (apar0>0)hptlx[0]->Fill((1+3.9*tk->error(0)/apar0)*tk->pt());      
      hcharge[1]->Fill(tk->charge()); 
      if ( tk->charge() != 0 ) {
	heta[1]->Fill(tk->eta());      
	hphi[1]->Fill(tk->phi()); 
	hetaphi[1]->Fill(tk->phi(),tk->eta()); 
	hptphi[1]->Fill(tk->pt(),tk->phi()); 
	hpteta[1]->Fill(tk->pt(),tk->eta()); 
	hnhit[1]->Fill(tk->numberOfValidHits()); 
	hd0[0]->Fill(tk->d0()); 
        if (!recoBeamSpotHandle.failedToGet()){
	  hdr[0]->Fill(tk->dxy(beamSpot.position()));	
	  hdrphi[0]->Fill(tk->phi(),tk->dxy(beamSpot.position())); 
	} 
	hd0phi[0]->Fill(tk->phi(),tk->d0()); 
	hdz[0]->Fill(tk->dz()); 
	hdzeta[0]->Fill(tk->eta(),tk->dz());
	herr0[0]->Fill(tk->error(0)); 
	cand2=cand;
	++cand2;
	for (; cand2!=l2mucands->end(); cand2++) {
	  TrackRef tk2=cand2->get<TrackRef>();
	  if ( tk->charge()*tk2->charge() == -1 ){
	    double mass=(cand->p4()+cand2->p4()).M();
	    hdimumass[0]->Fill(mass);
	  }
	}
      } else LogWarning("HLTMonMuon")<<"stop filling candidate with update@Vtx failure";
    }
  }
  if (!l3mucands.failedToGet()) {
    LogDebug("HLTMuonDQMSource") << " filling L3 stuff " << endl;
    hNMu[2]->Fill(l3mucands->size());
    Handle<reco::IsoDepositMap> l3depMap;
    iEvent.getByLabel (l3isolationTag_,l3depMap);
    for (cand=l3mucands->begin(); cand!=l3mucands->end(); ++cand) {
      TrackRef tk = cand->get<TrackRef>();
      if (!l3depMap.failedToGet()) {
	if ( l3depMap->contains(tk.id()) ){
	  reco::IsoDepositMap::value_type calDeposit= (*l3depMap)[tk];
	  double dephlt = calDeposit.depositWithin(coneSize_);
	  hiso[1]->Fill(dephlt);
	}
      }
      // eta cut
      hpt[2]->Fill(tk->pt());      
      double apar0 = fabs(tk->parameter(0));
      if (apar0>0)hptlx[1]->Fill((1+2.2*tk->error(0)/apar0)*tk->pt());      
      heta[2]->Fill(tk->eta());      
      hphi[2]->Fill(tk->phi()); 
      hetaphi[2]->Fill(tk->phi(),tk->eta()); 
      hptphi[2]->Fill(tk->pt(),tk->phi()); 
      hpteta[2]->Fill(tk->pt(),tk->eta()); 
      hnhit[2]->Fill(tk->numberOfValidHits()); 
      hd0[1]->Fill(tk->d0()); 
      if (!recoBeamSpotHandle.failedToGet()) {
	hdr[1]->Fill(tk->dxy(beamSpot.position()));
	hdrphi[1]->Fill(tk->phi(),tk->dxy(beamSpot.position())); 
      }
      hd0phi[1]->Fill(tk->phi(),tk->d0()); 
      hdz[1]->Fill(tk->dz()); 
      hdzeta[1]->Fill(tk->eta(),tk->dz());
      herr0[1]->Fill(tk->error(0)); 
      hcharge[2]->Fill(tk->charge()); 
      cand2=cand;
      ++cand2;

      for (; cand2!=l3mucands->end(); cand2++) {
	TrackRef tk2=cand2->get<TrackRef>();
	if ( tk->charge()*tk2->charge() == -1 ){
	  double mass=(cand->p4()+cand2->p4()).M();
	  hdimumass[1]->Fill(mass);
	  cout << "mass = " << mass << " " << cand->p4().M() << " " << cand2->p4().M() <<endl;
	}
      }
      if( tk->seedRef().castTo<Ref<L3MuonTrajectorySeedCollection> >().isAvailable() ) {
        TrackRef l2tk = tk->seedRef().castTo<Ref<L3MuonTrajectorySeedCollection> >()->l2Track();
        if(tk->pt()*l2tk->pt() != 0 )hptres[1]->Fill(1/tk->pt() - 1/l2tk->pt());
        hetares[1]->Fill(tk->eta()-l2tk->eta());
        hetareseta[1]->Fill(tk->eta(),tk->eta()-l2tk->eta());
        hphires[1]->Fill(tk->phi()-l2tk->phi());
        double dphi=tk->phi()-l2tk->phi();
        if (dphi>TMath::TwoPi())dphi-=2*TMath::TwoPi();
        else if (dphi<-TMath::TwoPi()) dphi+=TMath::TwoPi();
        hphiresphi[1]->Fill(tk->phi(),dphi);
      }
    }
  }  
}




//--------------------------------------------------------
void HLTMuonDQMSource::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				    const EventSetup& context) {
}
//--------------------------------------------------------
void HLTMuonDQMSource::endRun(const Run& r, const EventSetup& context){
}
//--------------------------------------------------------
void HLTMuonDQMSource::endJob(){
   LogInfo("HLTMonMuon") << "analyzed " << counterEvt_ << " events";
   cout << "analyzed = " << counterEvt_ << " , triggered = " << nTrig_ << endl;
 
   //  if (outputFile_.size() != 0 && dbe_)
   //  dbe->save(outputFile_);
 
   return;
}
