// -*- C++ -*-
//
// Package:    HLTMuonL1DQMSource
// Class:      HLTMuonL1DQMSource
// 
/**\class HLTMuonL1DQMSource 

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Muriel VANDER DONCKT *:0
//         Created:  Wed Dec 12 09:55:42 CET 2007
// $Id: HLTMuonL1DQMSource.cc,v 1.1 2008/06/25 10:46:57 muriel Exp $
//
//



#include "DQM/HLTEvF/interface/HLTMuonL1DQMSource.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "TMath.h" 


using namespace std;
using namespace edm;
using namespace l1extra;
//
// constructors and destructor
//
HLTMuonL1DQMSource::HLTMuonL1DQMSource( const edm::ParameterSet& parameters_ ) :counterEvt_(0)

{
  verbose_ = parameters_.getUntrackedParameter < bool > ("verbose", false);
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","HLT/HLTMuon");
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  level_ = parameters_.getUntrackedParameter<int>("Level",2);
  l1muCollectionTag_ = parameters_.getUntrackedParameter<InputTag>("L1MuonTag",edm::InputTag("hltL1extraParticles"));

   dbe_ = 0 ;
   if (parameters_.getUntrackedParameter < bool > ("DQMStore", false)) {
     dbe_ = Service < DQMStore > ().operator->();
     dbe_->setVerbose(0);
   }
 
   outputFile_ =
       parameters_.getUntrackedParameter < std::string > ("outputFile", "");
   if (outputFile_.size() != 0 && verbose_) {
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
     dbe_->setCurrentFolder("HLT/HLTMonMuon");
   }


}


HLTMuonL1DQMSource::~HLTMuonL1DQMSource()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}

//--------------------------------------------------------
void HLTMuonL1DQMSource::beginJob(){

 
   if (dbe_) {
     dbe_->setCurrentFolder(monitorName_);
     if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
     if (verbose_)cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;
     
     
     /// book some histograms here
     const int NBINS = 100; 

     // create and cd into new folder
     char name[512], title[512];
     sprintf(name,"Level%i",level_);
     dbe_->setCurrentFolder(monitorName_+name);
     hl1quality = dbe_->book1D("h1L1Quality","GMT quality Flag", 8, 0., 8.);
     sprintf(name,"HLTMuonL%i_NMu",level_);
     sprintf(title,"L%i number of muons",level_);
     hNMu = dbe_->book1D(name,title, 5, 0., 5.);
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
     sprintf(name,"HLTMuonL%i_charge",level_);
     sprintf(title,"L%i Muon Charge",level_);         
     hcharge  = dbe_->book1D(name,title, 3, -1.5, 1.5);
     hcharge->setAxisTitle("Charge", 1);
     if(verbose_)dbe_->showDirStructure();
       
     // Muon det id is 2 pushed in bits 28:31
     const unsigned int detector_id = 2<<28;
     dbe_->tagContents(monitorName_, detector_id);
   } 
}

//--------------------------------------------------------
void HLTMuonL1DQMSource::beginRun(const edm::Run& r, const EventSetup& context) {
  // reset all me's
  vector<MonitorElement*> AllME=dbe_->getAllContents(monitorName_);
  vector<MonitorElement*>::iterator me=AllME.begin();
  for ( ; me != AllME.end() ; ++me ){
    (*me)->Reset();
  }
}

//--------------------------------------------------------
void HLTMuonL1DQMSource::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				      const EventSetup& context) {
  
}

// ----------------------------------------------------------
void HLTMuonL1DQMSource::analyze(const Event& iEvent, 
			 const EventSetup& iSetup )
{  
  if ( !dbe_) return;
  counterEvt_++;
  if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0) return;
  if (verbose_)cout << " processing conterEvt_: " << counterEvt_ <<endl;

  edm::Handle<L1MuonParticleCollection> muColl;
  iEvent.getByLabel(l1muCollectionTag_, muColl);
  if (!muColl.failedToGet()){
    L1MuonParticleCollection::const_iterator l1ref;
    L1MuonParticleRef::key_type l1ParticleIndex = 0;
    hNMu->Fill(muColl->size());
    
    for(l1ref = muColl->begin(); l1ref != muColl->end(); ++l1ref,++l1ParticleIndex) {    
      hcharge->Fill(l1ref->charge()); 
      hpt->Fill(l1ref->pt());
      hphi->Fill(l1ref->phi());
      heta->Fill(l1ref->eta());
      hetaphi->Fill(l1ref->phi(),l1ref->eta());
      hptphi->Fill(l1ref->pt(),l1ref->phi());
      hpteta->Fill(l1ref->pt(),l1ref->eta());
      hl1quality->Fill(l1ref->gmtMuonCand().quality());
    }
  }
}



//--------------------------------------------------------
void HLTMuonL1DQMSource::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				    const EventSetup& context) {
}
//--------------------------------------------------------
void HLTMuonL1DQMSource::endRun(const Run& r, const EventSetup& context){
}
//--------------------------------------------------------
void HLTMuonL1DQMSource::endJob(){
   LogInfo("HLTMonMuon") << "analyzed " << counterEvt_ << " events";
 
   if (outputFile_.size() != 0 && dbe_)
    dbe_->save(outputFile_);
 
   return;
}
