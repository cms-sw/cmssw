// -*- C++ -*-
//
// Package:    HLTMuonIsoDQMSource
// Class:      HLTMuonIsoDQMSource
// 
/**\class HLTMuonIsoDQMSource 

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Muriel VANDER DONCKT *:0
//         Created:  Wed Dec 12 09:55:42 CET 2007
// $Id: HLTMuonIsoDQMSource.cc,v 1.1 2008/06/25 10:46:57 muriel Exp $
//
//



#include "DQM/HLTEvF/interface/HLTMuonIsoDQMSource.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include "TMath.h" 


using namespace std;
using namespace edm;
using namespace reco;
//
// constructors and destructor
//
HLTMuonIsoDQMSource::HLTMuonIsoDQMSource( const edm::ParameterSet& ps ) :counterEvt_(0)

{
  parameters_ = ps;
  verbose_ = parameters_.getUntrackedParameter < bool > ("verbose", false);
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","HLT/HLTMuon");
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  coneSize_ = parameters_.getUntrackedParameter<double>("coneSize", 0.24);
  candCollectionTag_ = parameters_.getUntrackedParameter<InputTag>("CandMuonTag",edm::InputTag("hltL2MuonCandidates"));
  isolationTag_ = parameters_.getUntrackedParameter<InputTag>("IsolationTag",edm::InputTag("hltL2MuonIsolations"));
  level_ = parameters_.getUntrackedParameter<int>("Level", 2);
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
     parameters_.getUntrackedParameter < bool > ("disableROOToutput", true);
   if (disable) {
     outputFile_ = "";
   }
 
   if (dbe_ != NULL) {
     dbe_->setCurrentFolder("HLT/HLTMuon");
   }


}


HLTMuonIsoDQMSource::~HLTMuonIsoDQMSource()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void HLTMuonIsoDQMSource::beginJob(){

 
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
     sprintf(name,"HLTMuonL%i_iso",level_);
     if (level_==2)sprintf(title,"L%i Muon Energy in Isolation cone",level_);         
     else if (level_==3)sprintf(title,"L%i Muon SumPt in Isolation cone",level_);               
     hiso  = dbe_->book1D(name,title, NBINS, 0., 10./(level_-2));
     if ( level_==2)hiso->setAxisTitle("Calo Energy in Iso Cone (GeV)", 1);
     else if ( level_==3)hiso->setAxisTitle("Sum Pt in Iso Cone (GeV)", 1);

     if(verbose_)dbe_->showDirStructure();
  
     // Muon det id is 2 pushed in bits 28:31
     const unsigned int detector_id = 2<<28;
     dbe_->tagContents(monitorName_, detector_id);
   } 
}

//--------------------------------------------------------
void HLTMuonIsoDQMSource::beginRun(const edm::Run& r, const EventSetup& context) {
  // reset all me's
  vector<MonitorElement*> AllME=dbe_->getAllContents(monitorName_);
  vector<MonitorElement*>::iterator me=AllME.begin();
  for ( ; me != AllME.end() ; ++me ){
    (*me)->Reset();
  }

}

//--------------------------------------------------------
void HLTMuonIsoDQMSource::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				      const EventSetup& context) {
  
}

// ----------------------------------------------------------
void HLTMuonIsoDQMSource::analyze(const Event& iEvent, 
			 const EventSetup& iSetup )
{  
  if ( !dbe_) return;
  counterEvt_++;
  if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0) return;
  if (verbose_)cout << " processing conterEvt_: " << counterEvt_ <<endl;

  Handle<RecoChargedCandidateCollection> mucands;
  iEvent.getByLabel (candCollectionTag_,mucands);
  RecoChargedCandidateCollection::const_iterator cand; 

  if (!mucands.failedToGet()) {
    Handle<reco::IsoDepositMap> depMap;
    iEvent.getByLabel (isolationTag_,depMap);
    for (cand=mucands->begin(); cand!=mucands->end(); ++cand) {
      TrackRef tk = cand->get<TrackRef>();
      if (!depMap.failedToGet()) {
	  if (verbose_)cout << " filling  Iso stuff " << endl;
	  if ( depMap->contains(tk.id()) ){
	    reco::IsoDepositMap::value_type calDeposit= (*depMap)[tk];
	    double dephlt = calDeposit.depositWithin(coneSize_);
	    hiso->Fill(dephlt);
	  }
      }
    }
  }
}




//--------------------------------------------------------
void HLTMuonIsoDQMSource::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				    const EventSetup& context) {
}
//--------------------------------------------------------
void HLTMuonIsoDQMSource::endRun(const Run& r, const EventSetup& context){
}
//--------------------------------------------------------
void HLTMuonIsoDQMSource::endJob(){
   LogInfo("HLTMonMuon") << "analyzed " << counterEvt_ << " events";
   if (outputFile_.size() != 0 && dbe_)
    dbe_->save(outputFile_);
   return;
}
