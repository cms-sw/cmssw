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
// $Id: HLTMuonDQMSource.cc,v 1.4 2008/02/12 17:36:49 muriel Exp $
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
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "TMath.h" 


using namespace std;
using namespace edm;
using namespace reco;
//
// constructors and destructor
//
HLTMuonDQMSource::HLTMuonDQMSource( const edm::ParameterSet& ps ) :counterEvt_(0)

{
  parameters_ = ps;
  verbose_ = parameters_.getUntrackedParameter < bool > ("verbose", false);
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","HLT/HLTMonMuon");
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  coneSize_ = parameters_.getUntrackedParameter<double>("coneSize", 0.24);
  l2collectionTag_ = parameters_.getUntrackedParameter<InputTag>("l2MuonTag",edm::InputTag("hltL2MuonCandidates"));
  l3collectionTag_ = parameters_.getUntrackedParameter<InputTag>("l3MuonTag",edm::InputTag("hltL3MuonCandidates"));
  l3linksTag_ = parameters_.getUntrackedParameter<InputTag>("l3MuonLinksTag",edm::InputTag("hltL3Muons"));
  l2isolationTag_ = parameters_.getUntrackedParameter<InputTag>("l2IsolationTag",edm::InputTag("hltL2MuonIsolations"));
  l3isolationTag_ = parameters_.getUntrackedParameter<InputTag>("l3IsolationTag",edm::InputTag("hltL3MuonIsolations"));

   dbe_ = 0 ;
   if (parameters_.getUntrackedParameter < bool > ("DQMStore", false)) {
     dbe_ = Service < DQMStore > ().operator->();
     dbe_->setVerbose(0);
   }
 
   outputFile_ =
       parameters_.getUntrackedParameter < std::string > ("outputFile", "");
   if (outputFile_.size() != 0) {
     std::cout << "Muon HLT Monitoring histograms will be saved to " 
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
     dbe_->setCurrentFolder("HLT/HLTMonElectron");
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
     dbe_->setCurrentFolder("monitorName_");
     if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
     if (verbose_)cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;
     
     
     /// book some histograms here
     const int NBINS = 100; XMIN = 0; XMAX = 50;

     // create and cd into new folder
     char name[512], title[512];
     for ( int level = 2 ; level < 4 ; ++ level ) {
       sprintf(name,"Level%i",level);
       dbe_->setCurrentFolder(monitorName_+name);
       sprintf(name,"HLTMuonL%i_NMu",level);
       sprintf(title,"L%i number of muons",level);
       hNMu[level-2] = dbe_->book1D(name,title, 5, 0., 5.);
       hNMu[level-2]->setAxisTitle("Number of muons", 1);
       sprintf(name,"HLTMuonL%i_pt",level);
       sprintf(title,"L%i Pt",level);
       hpt[level-2] = dbe_->book1D(name,title, NBINS, 0., 100);
       hpt[level-2]->setAxisTitle("Pt", 1);
       sprintf(name,"HLTMuonL%i_ptlx",level);
       sprintf(title,"L%i Muon 90 percent efficiency Pt",level);
       hptlx[level-2] = dbe_->book1D(name,title, NBINS, 0., 100);
       hptlx[level-2]->setAxisTitle("90% efficiency Pt", 1);
       sprintf(name,"HLTMuonL%i_eta",level);
       sprintf(title,"L%i Muon #eta",level);
       heta[level-2] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
       heta[level-2]->setAxisTitle("#eta", 1);
       sprintf(name,"HLTMuonL%i_phi",level);
       sprintf(title,"L%i Muon #phi",level);
       hphi[level-2] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
       hphi[level-2]->setAxisTitle("#phi", 1);
       sprintf(name,"HLTMuonL%i_etaphi",level);
       sprintf(title,"L%i Muon #eta vs #phi",level);
       hetaphi[level-2] = dbe_->book2D(name,title, NBINS, -3.15, 3.15,NBINS,-2.5, 2.5);
       hetaphi[level-2]->setAxisTitle("#phi", 1);
       hetaphi[level-2]->setAxisTitle("#eta", 2); 
       sprintf(name,"HLTMuonL%i_ptphi",level);
       sprintf(title,"L%i Muon pt vs #phi",level);         
       hptphi[level-2] = dbe_->book2D(name,title, NBINS, 0., 100.,NBINS,-3.15, 3.15);
       hptphi[level-2]->setAxisTitle("pt", 1);
       hptphi[level-2]->setAxisTitle("#phi", 2);
       sprintf(name,"HLTMuonL%i_pteta",level);
       sprintf(title,"L%i Muon pt vs #eta",level);         
       hpteta[level-2] = dbe_->book2D(name,title, NBINS, 0., 100.,NBINS,-2.5, 2.5);
       hpteta[level-2]->setAxisTitle("pt", 1);
       hpteta[level-2]->setAxisTitle("#eta", 2);
       sprintf(name,"HLTMuonL%i_dr",level);
       sprintf(title,"L%i Muon radial impact",level);         
       hdr[level-2] = dbe_->book1D(name,title, NBINS, -0.3, 0.3);
       hdr[level-2]->setAxisTitle("R Impact (cm)", 1);
       sprintf(name,"HLTMuonL%i_dz",level);
       sprintf(title,"L%i Muon Z impact",level);         
       hdz[level-2] = dbe_->book1D(name,title, NBINS, -25., 25.);
       hdz[level-2]->setAxisTitle("Z impact (cm)", 1);
       sprintf(name,"HLTMuonL%i_err0",level);
       sprintf(title,"L%i Muon Error on Pt",level);         
       herr0[level-2] = dbe_->book1D(name,title,NBINS, 0., 0.03);
       herr0[level-2]->setAxisTitle("Error on Pt", 1);
       sprintf(name,"HLTMuonL%i_nhit",level);
       sprintf(title,"L%i Number of Valid Hits",level);         
       hnhit[level-2] = dbe_->book1D(name,title, NBINS, 0., 100.);
       hnhit[level-2]->setAxisTitle("Number of Valid Hits", 1);
       sprintf(name,"HLTMuonL%i_charge",level);
       sprintf(title,"L%i Muon Charge",level);         
       hcharge[level-2]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
       hcharge[level-2]->setAxisTitle("Charge", 1);
       sprintf(name,"HLTMuonL%i_iso",level);
       if (level==2)sprintf(title,"L%i Muon Energy in Isolation cone",level);         
       else if (level==3)sprintf(title,"L%i Muon SumPt in Isolation cone",level);               
       hiso[level-2]  = dbe_->book1D(name,title, NBINS, 0., 10./(level-1));
       if ( level==2)hiso[level-2]->setAxisTitle("Calo Energy in Iso Cone (GeV)", 1);
       else if ( level==3)hiso[level-2]->setAxisTitle("Sum Pt in Iso Cone (GeV)", 1);
       sprintf(name,"HLTMuonL%i_DiMuMass",level);
       sprintf(title,"L%i Opposite charge DiMuon invariant Mass",level);         
       hdimumass[level-2]   = dbe_->book1D(name,title, NBINS, 0., 150.);
       hdimumass[level-2]->setAxisTitle("Di Muon Invariant Mass (GeV)");
       sprintf(name,"HLTMuonL%i_drphi",level);
       sprintf(title,"L%i #Deltar vs #phi",level);         
       hdrphi[level-2] = dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
       hdrphi[level-2]->setAxisTitle("#phi", 1);
       hdrphi[level-2]->setAxisTitle("#Deltar", 2);
       sprintf(name,"HLTMuonL%i_dzeta",level);
       sprintf(title,"L%i #Deltaz vs #eta",level);         
       hdzeta[level-2] = dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
       hdzeta[level-2]->setAxisTitle("#eta", 1);
       hdzeta[level-2]->setAxisTitle("#Deltaz", 2);
       if (level == 2 ) {
	 hL2ptres = dbe_->book1D("HLTMuonL2_ptres", "L2 Muon 1/Pt - L3Muon1/Pt", NBINS, -0.1, 0.1);
	 hL2ptres->setAxisTitle("1/PtL2-1/PtL3", 1);
	 hL2etares = dbe_->book1D("HLTMuonL2_etares", "L2 Muon #Delta#eta (wrt L3)", NBINS, -0.1, 0.1);
	 hL2etares->setAxisTitle("#Delta#eta", 1);
	 hL2phires = dbe_->book1D("HLTMuonL2_phires", "L2 Muon #Delta#phi (wrt L3)", NBINS, -0.1, 0.1);
	 hL2phires->setAxisTitle("#Delta#phi", 1);
	 hL2phiresphi = dbe_->bookProfile("HLTMuonL2_phiresphi", "L2 Muon #Delta#phi vs #phi", NBINS, -3.15, 3.15,1,-999.,999.,"s");
	 hL2phiresphi->setAxisTitle("<#Delta#phi>", 2);
	 hL2phiresphi->setAxisTitle("#phi", 1);
	 hL2etareseta = dbe_->bookProfile("HLTMuonL2_etareseta", "L2 Muon #Delta#eta vs #eta", NBINS,-2.5, 2.5,1,-999.,999.,"s");
	 hL2etareseta->setAxisTitle("<#Delta#eta>", 2);
	 hL2etareseta->setAxisTitle("#eta", 1);
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
  if (verbose_)cout << " processing conterEvt_: " << counterEvt_ <<endl;

  Handle<RecoChargedCandidateCollection> l2mucands, l3mucands;
  iEvent.getByLabel (l2collectionTag_,l2mucands);
  RecoChargedCandidateCollection::const_iterator cand,cand2;

  if (!l2mucands.failedToGet()) {
     if (verbose_)cout << " filling L2 stuff " << endl;
    Handle<MuIsoDepositAssociationMap> l2depMap;
    iEvent.getByLabel (l2isolationTag_,l2depMap);
    hNMu[0]->Fill(l2mucands->size());
    for (cand=l2mucands->begin(); cand!=l2mucands->end(); ++cand) {
      TrackRef tk = cand->get<TrackRef>();
      if (!l2depMap.failedToGet()) {
	  if (verbose_)cout << " filling L2 Iso stuff " << endl;
	  MuIsoDepositAssociationMap::const_iterator i = l2depMap->find(tk);
	  if ( i != l2depMap->end() ){
	    MuIsoDepositAssociationMap::result_type calDeposit= i->val;
	    double dephlt = calDeposit.depositWithin(coneSize_);
	    hiso[0]->Fill(dephlt);
	  } else LogWarning("HLTMonMuon") << "No calo iso deposit corresponding to tk";
      }
    
      // eta cut
      hpt[0]->Fill(tk->pt());      
      double apar0 = fabs(tk->parameter(0));
      if (apar0>0)hptlx[0]->Fill((1+3.9*tk->error(0)/apar0)*tk->pt());      
      hcharge[0]->Fill(tk->charge()); 
      if ( tk->charge() != 0 ) {
	heta[0]->Fill(tk->eta());      
	hphi[0]->Fill(tk->phi()); 
	hetaphi[0]->Fill(tk->phi(),tk->eta()); 
	hptphi[0]->Fill(tk->pt(),tk->phi()); 
	hpteta[0]->Fill(tk->pt(),tk->eta()); 
	hnhit[0]->Fill(tk->numberOfValidHits()); 
	hdr[0]->Fill(tk->d0()); 
	hdz[0]->Fill(tk->dz()); 
	hdrphi[0]->Fill(tk->phi(),tk->d0()); 
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
  iEvent.getByLabel (l3collectionTag_,l3mucands);
  if (!l3mucands.failedToGet()) {
    if (verbose_)cout << " filling L3 stuff " << endl;
    hNMu[1]->Fill(l3mucands->size());
    Handle<MuIsoDepositAssociationMap> l3depMap;
    iEvent.getByLabel (l3isolationTag_,l3depMap);
    for (cand=l3mucands->begin(); cand!=l3mucands->end(); ++cand) {
      TrackRef tk = cand->get<TrackRef>();
      if (!l3depMap.failedToGet()) {
	MuIsoDepositAssociationMap::const_iterator i = l3depMap->find(tk);
	if ( i != l3depMap->end() ) {
	  MuIsoDepositAssociationMap::result_type calDeposit= i->val;
	  double dephlt = calDeposit.depositWithin(coneSize_);
	  hiso[1]->Fill(dephlt);
	  } else LogWarning("HLTMonMuon") << "No track iso deposit corresponding to tk";
      }
      // eta cut
      hpt[1]->Fill(tk->pt());      
      double apar0 = fabs(tk->parameter(0));
      if (apar0>0)hptlx[1]->Fill((1+2.2*tk->error(0)/apar0)*tk->pt());      
      heta[1]->Fill(tk->eta());      
      hphi[1]->Fill(tk->phi()); 
      hetaphi[1]->Fill(tk->phi(),tk->eta()); 
      hptphi[1]->Fill(tk->pt(),tk->phi()); 
      hpteta[1]->Fill(tk->pt(),tk->eta()); 
      hnhit[1]->Fill(tk->numberOfValidHits()); 
      hdr[1]->Fill(tk->d0()); 
      hdz[1]->Fill(tk->dz()); 
      hdrphi[1]->Fill(tk->phi(),tk->d0()); 
      hdzeta[1]->Fill(tk->eta(),tk->dz());
      herr0[1]->Fill(tk->error(0)); 
      hcharge[1]->Fill(tk->charge()); 
      cand2=cand;
      ++cand2;

      for (; cand2!=l3mucands->end(); cand2++) {
	TrackRef tk2=cand2->get<TrackRef>();
	if ( tk->charge()*tk2->charge() == -1 ){
	  double mass=(cand->p4()+cand2->p4()).M();
	  hdimumass[1]->Fill(mass);
	}
      }
      Handle<MuonTrackLinksCollection> mulinks; 
      iEvent.getByLabel (l3linksTag_,mulinks);
      if (!mulinks.failedToGet()) {
	TrackRef l2tk;
	MuonTrackLinksCollection::const_iterator l3muon;
	for ( l3muon=mulinks->begin(); l3muon != mulinks->end();++l3muon){
	  if ( l3muon->globalTrack() == tk ) {
	    l2tk= l3muon->standAloneTrack();
	    if(tk->pt()*l2tk->pt() != 0 )hL2ptres->Fill(1/tk->pt() - 1/l2tk->pt());
	    hL2etares->Fill(tk->eta()-l2tk->eta());
	    hL2etareseta->Fill(tk->eta(),tk->eta()-l2tk->eta());
	    hL2phires->Fill(tk->phi()-l2tk->phi());
	    double dphi=tk->phi()-l2tk->phi();
	    if (dphi>TMath::TwoPi())dphi-=2*TMath::TwoPi();
	    else if (dphi<-TMath::TwoPi()) dphi+=TMath::TwoPi();
	    hL2phiresphi->Fill(tk->phi(),dphi);
	    break;
	  }
	}
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
 
   //  if (outputFile_.size() != 0 && dbe_)
   //  dbe->save(outputFile_);
 
   return;
}
