// -*- C++ -*-
//
// Package:    HLTMuonRecoDQMSource
// Class:      HLTMuonRecoDQMSource
// 
/**\class HLTMuonRecoDQMSource 

Description: <one line class summary>
Implementation:
<Notes on implementation>
*/
//
// Original Author:  Muriel VANDER DONCKT *:0
//         Created:  Wed Dec 12 09:55:42 CET 2007
// $Id: HLTMuonRecoDQMSource.cc,v 1.2 2008/10/16 16:41:29 hdyoo Exp $
//
//



#include "DQM/HLTEvF/interface/HLTMuonRecoDQMSource.h"
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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "TMath.h" 


using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;
//
// constructors and destructor
//
HLTMuonRecoDQMSource::HLTMuonRecoDQMSource( const edm::ParameterSet& parameters_ ) :counterEvt_(0)

{
  verbose_ = parameters_.getUntrackedParameter < bool > ("verbose", false);
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","HLT/HLTMuon");
  level_ = parameters_.getUntrackedParameter<int>("Level",2);
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  candCollectionTag_ = parameters_.getUntrackedParameter<InputTag>("CandMuonTag",edm::InputTag("hltL2MuonCandidates"));
  beamSpotTag_ = parameters_.getUntrackedParameter<InputTag>("BeamSpotTag",edm::InputTag("offlineBeamSpot"));
  l2seedscollectionTag_ = parameters_.getUntrackedParameter<InputTag>("l2MuonSeedTag",edm::InputTag("hltL2MuonSeeds"));

   dbe_ = 0 ;
   if (parameters_.getUntrackedParameter < bool > ("DQMStore", false)) {
     dbe_ = Service < DQMStore > ().operator->();
     dbe_->setVerbose(0);
   }
 
   outputFile_ =
       parameters_.getUntrackedParameter < std::string > ("outputFile", "");
   if (outputFile_.size() != 0) {
     if (verbose_) std::cout << "Muon HLT Monitoring histograms will be saved to " 
	       << outputFile_ << std::endl;
   }
   else {
     outputFile_ = "HLTMuonDQM.root";
   }
 
   bool disable =
     parameters_.getUntrackedParameter < bool > ("disableROOToutput", true);
   if (disable) {
     outputFile_ = "";
   }
 
   if (dbe_ != NULL) {
     dbe_->setCurrentFolder("HLT/HLTMuon");
   }


}


HLTMuonRecoDQMSource::~HLTMuonRecoDQMSource()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void HLTMuonRecoDQMSource::beginJob(){
 
   if (dbe_) {
     dbe_->setCurrentFolder("monitorName_");
     if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
     if (verbose_)cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;
     
     
     /// book some histograms here
     const int NBINS = 100; XMIN = 0; XMAX = 50;

     // create and cd into new folder
     char name[512], title[512];
     sprintf(name,"Level%i",level_);
     dbe_->setCurrentFolder(monitorName_+name);
     sprintf(name,"HLTMuonL%i_NMu",level_);
     sprintf(title,"L%i number of muons",level_);
     hNMu= dbe_->book1D(name,title, 5, 0., 5.);
     hNMu->setAxisTitle("Number of muons", 1);
     sprintf(name,"HLTMuonL%i_pt",level_);
     sprintf(title,"L%i Pt",level_);
     hpt = dbe_->book1D(name,title, NBINS, 0., 100);
     hpt->setAxisTitle("Pt", 1);
     sprintf(name,"HLTMuonL%i_eta",level_);
     sprintf(title,"L%i Muon #eta",level_);
     heta = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
     heta->setAxisTitle("#eta", 1);
     sprintf(name,"HLTMuonL%i_phi",level_);
     sprintf(title,"L%i Muon #phi",level_);
     hphi = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
     hphi->setAxisTitle("#phi", 1);
     sprintf(name,"HLTMuonL%i_etaphi",level_);
     sprintf(title,"L%i Muon #eta vs #phi",level_);
     hetaphi = dbe_->book2D(name,title, NBINS, -3.15, 3.15,NBINS,-2.5, 2.5);
     hetaphi->setAxisTitle("#phi", 1);
     hetaphi->setAxisTitle("#eta", 2); 
     sprintf(name,"HLTMuonL%i_ptphi",level_);
     sprintf(title,"L%i Muon pt vs #phi",level_);         
     hptphi = dbe_->book2D(name,title, NBINS, 0., 100.,NBINS,-3.15, 3.15);
     hptphi->setAxisTitle("pt", 1);
     hptphi->setAxisTitle("#phi", 2);
     sprintf(name,"HLTMuonL%i_pteta",level_);
     sprintf(title,"L%i Muon pt vs #eta",level_);         
     hpteta = dbe_->book2D(name,title, NBINS, 0., 100.,NBINS,-2.5, 2.5);
     hpteta->setAxisTitle("pt", 1);
     hpteta->setAxisTitle("#eta", 2);
     sprintf(name,"HLTMuonL%i_nhit",level_);
     sprintf(title,"L%i Number of Valid Hits",level_);         
     hnhit = dbe_->book1D(name,title, NBINS, 0., 100.);
     hnhit->setAxisTitle("Number of Valid Hits", 1);
     sprintf(name,"HLTMuonL%i_charge",level_);
     sprintf(title,"L%i Muon Charge",level_);         
     hcharge  = dbe_->book1D(name,title, 3, -1.5, 1.5);
     hcharge->setAxisTitle("Charge", 1);
     sprintf(name,"HLTMuonL%i_dr",level_);
     sprintf(title,"L%i Muon radial impact vs BeamSpot",level_);         
     hdr = dbe_->book1D(name,title, NBINS, -0.3, 0.3);
     hdr->setAxisTitle("R Impact (cm) vs BeamSpot", 1);
     sprintf(name,"HLTMuonL%i_d0",level_);
     sprintf(title,"L%i Muon radial impact vs (0,0)",level_);         
     hd0 = dbe_->book1D(name,title, NBINS, -0.3, 0.3);
     hd0->setAxisTitle("R Impact (cm) vs 0,0", 1);
     sprintf(name,"HLTMuonL%i_dz",level_);
     sprintf(title,"L%i Muon Z impact",level_);         
     hdz = dbe_->book1D(name,title, NBINS, -25., 25.);
     hdz->setAxisTitle("Z impact (cm)", 1);
     sprintf(name,"HLTMuonL%i_err0",level_);
     sprintf(title,"L%i Muon Error on Pt",level_);         
     herr0 = dbe_->book1D(name,title,NBINS, 0., 0.03);
     herr0->setAxisTitle("Error on Pt", 1);
     sprintf(name,"HLTMuonL%i_DiMuMass",level_);
     sprintf(title,"L%i Opposite charge DiMuon invariant Mass",level_);  
     hdimumass= dbe_->book1D(name,title, NBINS, 0., 150.);
     hdimumass->setAxisTitle("Di Muon Invariant Mass (GeV)");
     sprintf(name,"HLTMuonL%i_drphi",level_);
     sprintf(title,"L%i #Deltar vs #phi",level_);         
     hdrphi = dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
     hdrphi->setAxisTitle("#phi", 1);
     hdrphi->setAxisTitle("#Deltar", 2);
     sprintf(name,"HLTMuonL%i_d0phi",level_);
     sprintf(title,"L%i #Delta0 vs #phi",level_);         
     hd0phi = dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
     hd0phi->setAxisTitle("#phi", 1);
     hd0phi->setAxisTitle("#Delta0", 2);
     sprintf(name,"HLTMuonL%i_dzeta",level_);
     sprintf(title,"L%i #Deltaz vs #eta",level_);         
     hdzeta = dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
     hdzeta->setAxisTitle("#eta", 1);
     hdzeta->setAxisTitle("#Deltaz", 2);
     sprintf(name,"Level%i",level_-1);
     dbe_->setCurrentFolder(monitorName_+name);
     sprintf(name,"HLTMuonL%i_ptres",level_-1);
     sprintf(title,"L%iMuon1/Pt - L%iMuon1/Pt",level_-1,level_);         
     hptres = dbe_->book1D(name,title, NBINS, -0.1, 0.1);
     sprintf(title,"1/PtL%i - 1/PtL%i",level_-1,level_);         
     hptres->setAxisTitle(title, 1);
     sprintf(name,"HLTMuonL%i_etares",level_-1);
     sprintf(title,"L%i Muon #Delta#eta (wrt L%i)",level_-1,level_);         
     hetares =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
     hetares->setAxisTitle("#Delta#eta", 1);
     sprintf(name,"HLTMuonL%i_phires",level_-1);
     sprintf(title,"L%i Muon #Delta#phi (wrt L%i)",level_-1,level_);         
     hphires =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
     hphires->setAxisTitle("#Delta#phi", 1);
     sprintf(name,"HLTMuonL%i_phiresphi",level_-1);
     sprintf(title,"L%i Muon #Delta#phi vs #phi ",level_-1);         
     hphiresphi =dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
     hphiresphi->setAxisTitle("<#Delta#phi>", 2);
     hphiresphi->setAxisTitle("#phi", 1);
     sprintf(name,"HLTMuonL%i_etareseta",level_-1);
     sprintf(title,"L%i Muon #Delta#eta vs #eta ",level_-1);         
     hetareseta =dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
     hetareseta->setAxisTitle("<#Delta#eta>", 2);
     hetareseta->setAxisTitle("#eta", 1);
     if(verbose_)dbe_->showDirStructure();
     // Muon det id is 2 pushed in bits 28:31
     const unsigned int detector_id = 2<<28;
     dbe_->tagContents(monitorName_, detector_id);
   }
} 

//--------------------------------------------------------
void HLTMuonRecoDQMSource::beginRun(const edm::Run& r, const EventSetup& context) {
  // reset all me's
  vector<MonitorElement*> AllME=dbe_->getAllContents(monitorName_);
  vector<MonitorElement*>::iterator me=AllME.begin();
  for ( ; me != AllME.end() ; ++me ){
    (*me)->Reset();
  }
}

//--------------------------------------------------------
void HLTMuonRecoDQMSource::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				      const EventSetup& context) {
  
}

// ----------------------------------------------------------
void HLTMuonRecoDQMSource::analyze(const Event& iEvent, 
			 const EventSetup& iSetup )
{  
  if ( !dbe_) return;
  counterEvt_++;
  if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0) return;
  if (verbose_)cout << " processing conterEvt_: " << counterEvt_ <<endl;

  Handle<RecoChargedCandidateCollection> mucands;
  iEvent.getByLabel (candCollectionTag_,mucands);

  reco::BeamSpot beamSpot;
  Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpotTag_,recoBeamSpotHandle);
  if (!recoBeamSpotHandle.failedToGet())  beamSpot = *recoBeamSpotHandle;
  RecoChargedCandidateCollection::const_iterator cand,cand2;  
  if (!mucands.failedToGet()) {
    if (verbose_)cout << " filling Reco stuff " << endl;
    hNMu->Fill(mucands->size());
    for (cand=mucands->begin(); cand!=mucands->end(); ++cand) {
      TrackRef tk = cand->get<TrackRef>();      
      // eta cut
      hpt->Fill(tk->pt());      
      hcharge->Fill(tk->charge()); 
      if ( tk->charge() != 0 ) {
	heta->Fill(tk->eta());      
	hphi->Fill(tk->phi()); 
	hetaphi->Fill(tk->phi(),tk->eta()); 
	hptphi->Fill(tk->pt(),tk->phi()); 
	hpteta->Fill(tk->pt(),tk->eta()); 
	hnhit->Fill(tk->numberOfValidHits()); 
	hd0->Fill(tk->d0()); 
        if (!recoBeamSpotHandle.failedToGet()){
	  hdr->Fill(tk->dxy(beamSpot.position()));	
	  hdrphi->Fill(tk->phi(),tk->dxy(beamSpot.position())); 
	} 
	hd0phi->Fill(tk->phi(),tk->d0()); 
	hdz->Fill(tk->dz()); 
	hdzeta->Fill(tk->eta(),tk->dz());
	herr0->Fill(tk->error(0)); 
	cand2=cand;
	++cand2;
	for (; cand2!=mucands->end(); cand2++) {
	  TrackRef tk2=cand2->get<TrackRef>();
	  if ( tk->charge()*tk2->charge() == -1 ){
	    double mass=(cand->p4()+cand2->p4()).M();
	    hdimumass->Fill(mass);
	  }
	}
	if ( level_ == 3 ) {
	  TrackRef l2tk=tk->seedRef().castTo<Ref<L3MuonTrajectorySeedCollection> >()->l2Track();
	  if(tk->pt()*l2tk->pt() != 0 )hptres->Fill(1/tk->pt() - 1/l2tk->pt());
	  hetares->Fill(tk->eta()-l2tk->eta());
	  hetareseta->Fill(tk->eta(),tk->eta()-l2tk->eta());
	  hphires->Fill(tk->phi()-l2tk->phi());
	  double dphi=tk->phi()-l2tk->phi();
	  if (dphi>TMath::TwoPi())dphi-=2*TMath::TwoPi();
	  else if (dphi<-TMath::TwoPi()) dphi+=TMath::TwoPi();
	  hphiresphi->Fill(tk->phi(),dphi);
	} else {
	  Handle<L2MuonTrajectorySeedCollection> museeds;
	  iEvent.getByLabel (l2seedscollectionTag_,museeds);
	  if (!museeds.failedToGet()){ 
	    RefToBase<TrajectorySeed> seed=tk->seedRef();
	    L1MuonParticleRef l1ref;
	    for(uint iMuSeed=0 ; iMuSeed!=museeds->size(); ++iMuSeed){
	      Ref<L2MuonTrajectorySeedCollection> l2seed(museeds,iMuSeed);
	      if (l2seed.id()==seed.id() && l2seed.key()==seed.key()){
		l1ref = l2seed->l1Particle();
		break;
	      }
	    }
	    if( tk->pt()*l1ref->pt() != 0 )hptres->Fill(1/tk->pt() - 1/l1ref->pt());
	    hetares->Fill(tk->eta()-l1ref->eta());
	    hetareseta->Fill(tk->eta(),tk->eta()-l1ref->eta());
	    hphires->Fill(tk->phi()-l1ref->phi());
	    double dphi=tk->phi()-l1ref->phi();
	    if (dphi>TMath::TwoPi())dphi-=2*TMath::TwoPi();
	    else if (dphi<-TMath::TwoPi()) dphi+=TMath::TwoPi();
	    hphiresphi->Fill(tk->phi(),dphi);
	  }
	}
      } else LogWarning("HLTMonMuon")<<"stop filling candidate with update@Vtx failure";
    }
  }
}




//--------------------------------------------------------
void HLTMuonRecoDQMSource::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				    const EventSetup& context) {
}
//--------------------------------------------------------
void HLTMuonRecoDQMSource::endRun(const Run& r, const EventSetup& context){
}
//--------------------------------------------------------
void HLTMuonRecoDQMSource::endJob(){
   LogInfo("HLTMonMuon") << "analyzed " << counterEvt_ << " events";
 
   if (outputFile_.size() != 0 && dbe_)
    dbe_->save(outputFile_);
 
   return;
}
