// -*- C++ -*-
//
// Package:    HLTMuonDQMSource
// Class:      HLTMuonDQMSource
// 
/**\class HLTMuonDQMSource HLTMuonDQMSource.cc HLTriggerOffline/HLTMuonDQMSource/src/HLTMuonDQMSource.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Muriel VANDER DONCKT *:0
//         Created:  Wed Dec 12 09:55:42 CET 2007
// $Id: HLTMuonDQMSource.cc,v 1.1 2008/01/24 15:25:36 muriel Exp $
//
//



#include "DQM/HLTEvF/interface/HLTMuonDQMSource.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"




using namespace std;
using namespace edm;
using namespace reco;

//
// constructors and destructor
//
HLTMuonDQMSource::HLTMuonDQMSource( const edm::ParameterSet& ps ) :counterEvt_(0)

{
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  parameters_ = ps;
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","HLTMuon");
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  coneSize_ = parameters_.getUntrackedParameter<double>("coneSize", 0.24);
  l2collectionTag_ = parameters_.getUntrackedParameter<InputTag>("l2MuonTag",edm::InputTag("hltL2MuonCandidates"));
  l3collectionTag_ = parameters_.getUntrackedParameter<InputTag>("l3MuonTag",edm::InputTag("hltL3MuonCandidates"));
  l3linksTag_ = parameters_.getUntrackedParameter<InputTag>("l3MuonLinksTag",edm::InputTag("hltL3Muons"));
  l2isolationTag_ = parameters_.getUntrackedParameter<InputTag>("l2IsolationTag",edm::InputTag("hltL2MuonIsolations"));
  l3isolationTag_ = parameters_.getUntrackedParameter<InputTag>("l3IsolationTag",edm::InputTag("hltL3MuonIsolations"));

  cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;
 

  /// book some histograms here
  const int NBINS = 100; XMIN = 0; XMAX = 50;

  // create and cd into new folder

  dbe_->setCurrentFolder(monitorName_+"Level2/Objects");
  hL2NMu = dbe_->book1D("HLTMuonL2_NMu", "L2 Number of Muons", 5, 0., 5.);
  hL2NMu->setAxisTitle("Number of muons", 1);
  hL2pt = dbe_->book1D("HLTMuonL2_pt", "L2 Muon Pt", NBINS, 0., 100);
  hL2pt->setAxisTitle("Pt", 1);
  hL2ptlx = dbe_->book1D("HLTMuonL2_ptlx", "L2 Muon 90% efficiency Pt", NBINS, 0., 100);
  hL2ptlx->setAxisTitle("90% efficienct Pt", 1);
  hL2eta = dbe_->book1D("HLTMuonL2_eta", "L2 Muon #eta", NBINS, -2.5, 2.5);
  hL2eta->setAxisTitle("#eta", 1);
  hL2phi = dbe_->book1D("HLTMuonL2_phi", "L2 Muon #phi", NBINS, -3.15, 3.15);
  hL2phi->setAxisTitle("#phi", 1);
  hL2etaphi = dbe_->book2D("HLTMuonL2_etaphi", "L2 Muon #eta vs #phi", NBINS, -3.15, 3.15,NBINS,-2.5, 2.5);
  hL2etaphi->setAxisTitle("#phi", 1);
  hL2etaphi->setAxisTitle("#eta", 2);
  hL2dr = dbe_->book1D("HLTMuonL2_dr", "L2 Muon radial impact", NBINS, -0.5, 0.5);
  hL2dr->setAxisTitle("R Impact (cm)", 1);
  hL2dz = dbe_->book1D("HLTMuonL2_dz", "L2 Muon Z impact", NBINS, -25., 25.);
  hL2dz->setAxisTitle("Z impact (cm)", 1);
  hL2err0 = dbe_->book1D("HLTMuonL2_err0", "L2 Muon Error on Slope", NBINS, 0., 0.03);
  hL2err0->setAxisTitle("Error on Slope", 1);
  hL2nhit = dbe_->book1D("HLTMuonL2_nhit", "L2 Number of Valid Hits", NBINS, 0., 200.);
  hL2nhit->setAxisTitle("Number of Valid Hits", 1);
  hL2charge  = dbe_->book1D("HLTMuonL2_charge", "L2 Muon Charge", 3, -1.5, 1.5);
  hL2charge->setAxisTitle("Charge", 1);
  
  hL2iso = dbe_->book1D("HLTMuonL2_iso", "L2 Muon Energy in Isolation cone", NBINS, 0., 10.);
  hL2iso->setAxisTitle("Calo Energy in Iso Cone (GeV)", 1);
  hL2dimumass  = dbe_->book1D("HLTMuonL2_DiMuMass", "L2 Opposite charge DiMuon invariant Mass", NBINS, 0., 150.);
  hL2dimumass->setAxisTitle("Di Muon Invariant Mass (GeV)");
  hL2ptres = dbe_->book1D("HLTMuonL2_ptres", "L2 Muon 1/Pt - L3Muon1/Pt", NBINS, -0.1, 0.1);
  hL2ptres->setAxisTitle("1/PtL2-1/PtL3", 1);
  hL2etares = dbe_->book1D("HLTMuonL2_etares", "L2 Muon #Delta#eta (wrt L3)", NBINS, -0.1, 0.1);
  hL2etares->setAxisTitle("#Delta#eta", 1);
  hL2phires = dbe_->book1D("HLTMuonL2_phires", "L2 Muon #Delta#phi (wrt L3)", NBINS, -0.1, 0.1);
  hL2phires->setAxisTitle("#Delta#phi", 1);

  dbe_->setCurrentFolder(monitorName_+"Level3/Objects");
  hL3NMu = dbe_->book1D("HLTMuonL3_NMu", "L3 Number of Muons", 5, 0., 5.);
  hL3NMu->setAxisTitle("Number of muons", 1);
  hL3pt = dbe_->book1D("HLTMuonL3_pt", "L3 Muon Pt", NBINS, 0., 100);
  hL3pt->setAxisTitle("Pt", 1);
  hL3ptlx = dbe_->book1D("HLTMuonL3_ptlx", "L3 Muon 90% efficiency Pt", NBINS, 0., 100);
  hL3ptlx->setAxisTitle("90% efficiency Pt", 1);
  hL3eta = dbe_->book1D("HLTMuonL3_eta", "L3 Muon #eta", NBINS, -2.5, 2.5);
  hL3eta->setAxisTitle("#eta", 1);
  hL3phi = dbe_->book1D("HLTMuonL3_phi", "L3 Muon #phi", NBINS, -3.15, 3.15);
  hL3phi->setAxisTitle("#phi", 1);
  hL3etaphi = dbe_->book2D("HLTMuonL3_etaphi", "L3 Muon #eta vs #phi", NBINS, -3.15, 3.15,NBINS,-2.5, 2.5);
  hL3etaphi->setAxisTitle("#phi", 1);
  hL3etaphi->setAxisTitle("#eta", 2);
  hL3dr = dbe_->book1D("HLTMuonL3_dr", "L3 Muon radial impact", NBINS, -0.5, 0.5);
  hL3dr->setAxisTitle("R Impact (cm)", 1);
  hL3dz = dbe_->book1D("HLTMuonL3_dz", "L3 Muon Z impact", NBINS, -25., 25.);
  hL3dz->setAxisTitle("Z impact (cm)", 1);
  hL3err0 = dbe_->book1D("HLTMuonL3_err0", "L3 Muon Error on Slope", NBINS, 0., 0.01);
  hL3err0->setAxisTitle("Error on Slope", 1);
  hL3nhit = dbe_->book1D("HLTMuonL3_nhit", "L3 Number of Valid Hits", NBINS, 0., 200.);
  hL3nhit->setAxisTitle("Number of Valid Hits", 1);
  hL3iso = dbe_->book1D("HLTMuonL3_iso", "L3 Muon track pt in Isolation cone", NBINS, 0., 5.);
  hL3iso->setAxisTitle("Track Pt in Iso Cone (GeV)", 1);
  hL3dimumass  = dbe_->book1D("HLTMuonL3_DiMuMass", "L3 Opposite Charge DiMuon invariant Mass", NBINS, 0., 150.);
  hL3dimumass->setAxisTitle("Di Muon Invariant Mass (GeV)");
  hL3charge  = dbe_->book1D("HLTMuonL3_charge", "L3 Muon Charge", 3, -1.5, 1.5);
  hL3charge->setAxisTitle("Charge", 1);
 
  dbe_->showDirStructure();
  

  // Muon det id is 2 pushed in bits 28:31
  const unsigned int detector_id = 2<<28;
  dbe_->tagContents(monitorName_, detector_id);
  
}


HLTMuonDQMSource::~HLTMuonDQMSource()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void HLTMuonDQMSource::beginJob(const EventSetup& context){

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
  counterEvt_++;
  if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0) return;
  // cout << " processing conterEvt_: " << counterEvt_ <<endl;
  Handle<RecoChargedCandidateCollection> l2mucands, l3mucands;
  iEvent.getByLabel (l2collectionTag_,l2mucands);
  RecoChargedCandidateCollection::const_iterator cand,cand2;
  if (!l2mucands.failedToGet()) {
    Handle<MuIsoDepositAssociationMap> l2depMap;
    iEvent.getByLabel (l2isolationTag_,l2depMap);
    hL2NMu->Fill(l2mucands->size());
    for (cand=l2mucands->begin(); cand!=l2mucands->end(); ++cand) {
      TrackRef tk = cand->get<TrackRef>();
      if (!l2depMap.failedToGet()) {
	MuIsoDepositAssociationMap::result_type calDeposit= (*l2depMap)[tk];
	double dephlt = calDeposit.depositWithin(coneSize_);
	hL2iso->Fill(dephlt);
      }
      // eta cut
      hL2pt->Fill(tk->pt());      
      double apar0 = fabs(tk->parameter(0));
      if (apar0>0)hL2ptlx->Fill((1+3.9*tk->error(0)/apar0)*tk->pt());      
      hL2eta->Fill(tk->eta());      
      hL2phi->Fill(tk->phi()); 
      hL2etaphi->Fill(tk->phi(),tk->eta()); 
      hL2nhit->Fill(tk->numberOfValidHits()); 
      hL2dr->Fill(tk->d0()); 
      hL2dz->Fill(tk->dz()); 
      hL2err0->Fill(tk->error(0)); 
      hL2charge->Fill(tk->charge()); 
      cand2=cand;
      ++cand2;
      for (; cand2!=l2mucands->end(); cand2++) {
	if ( tk->charge() != cand2->get<TrackRef>()->charge() ){
	  double mass=(cand->p4()+cand2->p4()).M();
	  hL2dimumass->Fill(mass);
	}
      }
    }
  }
  iEvent.getByLabel (l3collectionTag_,l3mucands);
  if (!l3mucands.failedToGet()) {
    hL3NMu->Fill(l3mucands->size());
    Handle<MuIsoDepositAssociationMap> l3depMap;
    iEvent.getByLabel (l3isolationTag_,l3depMap);
    for (cand=l3mucands->begin(); cand!=l3mucands->end(); ++cand) {
      TrackRef tk = cand->get<TrackRef>();
      if (!l3depMap.failedToGet()) {
	MuIsoDepositAssociationMap::result_type calDeposit= (*l3depMap)[tk];
	double dephlt = calDeposit.depositWithin(coneSize_);
	hL3iso->Fill(dephlt);
      }
      // eta cut
      hL3pt->Fill(tk->pt());      
      double apar0 = fabs(tk->parameter(0));
      if (apar0>0)hL3ptlx->Fill((1+2.2*tk->error(0)/apar0)*tk->pt());      
      hL3eta->Fill(tk->eta());      
      hL3phi->Fill(tk->phi()); 
      hL3etaphi->Fill(tk->phi(),tk->eta()); 
      hL3nhit->Fill(tk->numberOfValidHits()); 
      hL3dr->Fill(tk->d0()); 
      hL3dz->Fill(tk->dz()); 
      hL3err0->Fill(tk->error(0)); 
      hL3charge->Fill(tk->charge()); 
      cand2=cand;
      ++cand2;
      for (; cand2!=l3mucands->end(); cand2++) {
	if ( tk->charge() != cand2->get<TrackRef>()->charge() ){
	  double mass=(cand->p4()+cand2->p4()).M();
	  hL3dimumass->Fill(mass);
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
	    hL2phires->Fill(tk->phi()-l2tk->phi());
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
}
