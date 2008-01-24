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
// $Id$
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
  cout << "Monitor name = " << monitorName_ << endl;
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;
 

  /// book some histograms here
  const int NBINS = 100; XMIN = 0; XMAX = 50;

  // create and cd into new folder

  dbe_->setCurrentFolder(monitorName_+"Level2/Objects");
  hL2NMu = dbe_->book1D("HLTMuonL2_NMu", "L2 Number of Muons", 5, 0., 5.);
  hL2NMu->setAxisTitle("Number of muons", 1);
  hL2pt = dbe_->book1D("HLTMuonL2_pt", "L2 Muon Pt", NBINS, 0., 100);
  hL2pt->setAxisTitle("Pt", 1);
  hL2highpt = dbe_->book1D("HLTMuonL2_highpt", "L2 Muon Pt", NBINS, 100., 1000.);
  hL2highpt->setAxisTitle("Pt", 1);
  hL2eta = dbe_->book1D("HLTMuonL2_eta", "L2 Muon #eta", NBINS, -2.5, 2.5);
  hL2eta->setAxisTitle("Eta", 1);
  hL2phi = dbe_->book1D("HLTMuonL2_phi", "L2 Muon #phi", NBINS, -3.15, 3.15);
  hL2phi->setAxisTitle("Phi", 1);
  hL2etaphi = dbe_->book2D("HLTMuonL2_etaphi", "L2 Muon #eta vs #phi", NBINS, -3.15, 3.15,NBINS,-2.5, 2.5);
  hL2etaphi->setAxisTitle("#phi", 1);
  hL2etaphi->setAxisTitle("#eta", 2);
  hL2dr = dbe_->book1D("HLTMuonL2_dr", "L2 Muon radial impact", NBINS, 0., 1.);
  hL2dr->setAxisTitle("R Impact (cm)", 1);
  hL2dz = dbe_->book1D("HLTMuonL2_dz", "L2 Muon Z impact", NBINS, -25., 25.);
  hL2dz->setAxisTitle("Z impact (cm)", 1);
  hL2err0 = dbe_->book1D("HLTMuonL2_err0", "L2 Muon Error on Slope", NBINS, 0., 1.);
  hL2err0->setAxisTitle("Error on Slope", 1);
  hL2nhit = dbe_->book1D("HLTMuonL2_nhit", "L2 Number of Valid Hits", NBINS, 0., 200.);
  hL2nhit->setAxisTitle("Number of Valid Hits", 1);
  
  hL2iso = dbe_->book1D("HLTMuonL2_iso", "L2 Muon Energy in Isolation cone", NBINS, 0., 1.);
  hL2iso->setAxisTitle("Calo Energy in Iso Cone (GeV)", 1);
  hL2dimumass  = dbe_->book1D("HLTMuonL2_DiMuMass", "L2 Opposite charge DiMuon invariant Mass", NBINS, 0., 150.);
  hL2dimumass->setAxisTitle("Di Muon Invariant Mass (GeV)");

  dbe_->setCurrentFolder(monitorName_+"Level3/Objects");
  hL3NMu = dbe_->book1D("HLTMuonL3_NMu", "L3 Number of Muons", 5, 0., 5.);
  hL3NMu->setAxisTitle("Number of muons", 1);
  hL3pt = dbe_->book1D("HLTMuonL3_pt", "L3 Muon Pt", NBINS, 0., 100);
  hL3pt->setAxisTitle("Pt", 1);
  hL3eta = dbe_->book1D("HLTMuonL3_eta", "L3 Muon #eta", NBINS, -2.5, 2.5);
  hL3eta->setAxisTitle("Eta", 1);
  hL3phi = dbe_->book1D("HLTMuonL3_phi", "L3 Muon #phi", NBINS, -3.15, 3.15);
  hL3phi->setAxisTitle("Phi", 1);
  hL3etaphi = dbe_->book2D("HLTMuonL3_etaphi", "L3 Muon #eta vs #phi", NBINS, -3.15, 3.15,NBINS,-2.5, 2.5);
  hL3etaphi->setAxisTitle("#phi", 1);
  hL3etaphi->setAxisTitle("#eta", 2);
  hL3highpt = dbe_->book1D("HLTMuonL3_highpt", "L3 Muon Pt", NBINS, 100., 1000.);
  hL3highpt->setAxisTitle("Pt", 1);
  hL3dr = dbe_->book1D("HLTMuonL3_dr", "L3 Muon radial impact", NBINS, 0., 1.);
  hL3dr->setAxisTitle("R Impact (cm)", 1);
  hL3dz = dbe_->book1D("HLTMuonL3_dz", "L3 Muon Z impact", NBINS, -25., 25.);
  hL3dz->setAxisTitle("Z impact (cm)", 1);
  hL3err0 = dbe_->book1D("HLTMuonL3_err0", "L3 Muon Error on Slope", NBINS, 0., 1.);
  hL3err0->setAxisTitle("Error on Slope", 1);
  hL3nhit = dbe_->book1D("HLTMuonL3_nhit", "L3 Number of Valid Hits", NBINS, 0., 200.);
  hL3nhit->setAxisTitle("Number of Valid Hits", 1);
  hL3iso = dbe_->book1D("HLTMuonL3_iso", "L3 Muon track pt in Isolation cone", NBINS, 0., 1.);
  hL3iso->setAxisTitle("Track Pt in Iso Cone (GeV)", 1);
  hL3dimumass  = dbe_->book1D("HLTMuonL3_DiMuMass", "L3 Opposite Charge DiMuon invariant Mass", NBINS, 0., 150.);
  hL3dimumass->setAxisTitle("Di Muon Invariant Mass (GeV)");
 
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
  iEvent.getByLabel ("hltL2MuonCandidates",l2mucands);
  RecoChargedCandidateCollection::const_iterator cand,cand2;
  if (!l2mucands.failedToGet()) {
    //    Handle<MuIsoDepositAssociationMap> depMap;
    //    iEvent.getByLabel (isoTag_,depMap);
    hL2NMu->Fill(l2mucands->size());
    for (cand=l2mucands->begin(); cand!=l2mucands->end(); ++cand) {
      TrackRef tk = cand->get<TrackRef>();
      // eta cut
      hL2pt->Fill(tk->pt());      
      hL2eta->Fill(tk->eta());      
      hL2phi->Fill(tk->phi()); 
      hL2etaphi->Fill(tk->phi(),tk->eta()); 
      hL2nhit->Fill(tk->numberOfValidHits()); 
      hL2dr->Fill(tk->d0()); 
      hL2dz->Fill(tk->dz()); 
      hL2err0->Fill(tk->error(0)); 
      //      MuIsoDepositAssociationMap::energy = (*depMap)[tk];
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
  iEvent.getByLabel ("hltL3MuonCandidates",l3mucands);
  if (!l3mucands.failedToGet()) {
    hL3NMu->Fill(l3mucands->size());
    for (cand=l3mucands->begin(); cand!=l3mucands->end(); ++cand) {
      TrackRef tk = cand->get<TrackRef>();
      // eta cut
      hL3pt->Fill(tk->pt());      
      hL3eta->Fill(tk->eta());      
      hL3phi->Fill(tk->phi()); 
      hL3etaphi->Fill(tk->phi(),tk->eta()); 
      hL3nhit->Fill(tk->numberOfValidHits()); 
      hL3dr->Fill(tk->d0()); 
      hL3dz->Fill(tk->dz()); 
      hL3err0->Fill(tk->error(0)); 
      cand2=cand;
      ++cand2;
      for (; cand2!=l3mucands->end(); cand2++) {
	if ( tk->charge() != cand2->get<TrackRef>()->charge() ){
	  double mass=(cand->p4()+cand2->p4()).M();
	  hL3dimumass->Fill(mass);
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
