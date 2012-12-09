/*
 * \file BeamSpotProblemMonitor.cc
 * \author Sushil S. Chauhan/UC Davis
 *        
 * $Date: 2012/05/22 19:44:12 $
 * $Revision: 1.1 $
 */




#include "DQM/BeamMonitor/plugins/BeamSpotProblemMonitor.h"
#include "DQMServices/Core/interface/QReport.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "RecoVertex/BeamSpotProducer/interface/BeamSpotOnlineProducer.h"                                                                             
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DQMServices/Core/interface/QReport.h"

#include <numeric>
#include <math.h>
#include <TMath.h>
#include <iostream>
#include <TStyle.h>

using namespace std;
using namespace edm;


#define buffTime (23)

//
// constructors and destructor
//
BeamSpotProblemMonitor::BeamSpotProblemMonitor( const ParameterSet& ps ) :
  Ntracks_(0), fitNLumi_(0), ALARM_ON_(false), BeamSpotStatus_(0), BeamSpotFromDB_(0){

  parameters_     = ps;
  monitorName_    = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  trkSrc_         = parameters_.getUntrackedParameter<InputTag>("pixelTracks");
  nCosmicTrk_     = parameters_.getUntrackedParameter<int>("nCosmicTrk");
  scalertag_      = parameters_.getUntrackedParameter<InputTag>("scalarBSCollection");
  intervalInSec_  = parameters_.getUntrackedParameter<int>("timeInterval",920);//40 LS X 23"
  debug_          = parameters_.getUntrackedParameter<bool>("Debug");
  onlineMode_     = parameters_.getUntrackedParameter<bool>("OnlineMode");
  alarmONThreshold_ = parameters_.getUntrackedParameter<int>("AlarmONThreshold");
  alarmOFFThreshold_= parameters_.getUntrackedParameter<int>("AlarmOFFThreshold");
  doTest_           =parameters_.getUntrackedParameter<bool>("doTest");

  dbe_            = Service<DQMStore>().operator->();
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;


  if (fitNLumi_ <= 0) fitNLumi_ = 1;
  lastlumi_ = 0;
  nextlumi_ = 0;
  processed_ = false;
}


BeamSpotProblemMonitor::~BeamSpotProblemMonitor() {
}


//--------------------------------------------------------
void BeamSpotProblemMonitor::beginJob() {


  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"FitFromScalars");

  int nbins = alarmOFFThreshold_;
  double hiRange = (alarmOFFThreshold_+0.5);  

  const int nvar_ = 1;
  string coord[nvar_] = {"BeamSpotStatus"};
  string label[nvar_] = {"BeamSpotStatus "};

  for (int i = 0; i < 1; i++) {
    dbe_->setCurrentFolder(monitorName_+"FitFromScalars");
    for (int ic=0; ic<nvar_; ++ic) {
      TString histName(coord[ic]);
      TString histTitle(coord[ic]);
      string ytitle("Problem (-1)  /  OK (1)");
      string xtitle("");
      string options("E1");
      bool createHisto = true;
     switch(i){

     case 0: 
        histName += "_lumi";
	xtitle = "Lumisection";

      if (createHisto) {
	hs[histName] = dbe_->book1D(histName,histTitle,40,0.5,40.5);
	hs[histName]->setAxisTitle(xtitle,1);
	hs[histName]->setAxisTitle(ytitle,2);

        histName += "_all";
        histTitle += " all";
        hs[histName] = dbe_->book1D(histName,histTitle,40,0.5,40.5);
        hs[histName]->getTH1()->SetBit(TH1::kCanRebin);
        hs[histName]->setAxisTitle(xtitle,1);
        hs[histName]->setAxisTitle(ytitle,2);

        }//create histo
      }
    }
  }

  BeamSpotError = dbe_->book1D("BeamSpotError","ERROR: Beamspot missing from scalars",20,0.5,20.5);
  BeamSpotError->setAxisTitle("# of consecutive LSs with problem",1);
  BeamSpotError->setAxisTitle("Problem with scalar BeamSpot",2);


  dbe_->setCurrentFolder(monitorName_+"FitFromScalars");



}

//--------------------------------------------------------
void BeamSpotProblemMonitor::beginRun(const edm::Run& r, const EventSetup& context) {


  if (debug_) {
    edm::LogInfo("BeamSpotProblemMonitor") << "TimeOffset = ";
  }

}

//--------------------------------------------------------
void BeamSpotProblemMonitor::beginLuminosityBlock(const LuminosityBlock& lumiSeg,
				       const EventSetup& context) {
  int nthlumi = lumiSeg.luminosityBlock();


  if (onlineMode_) {
    if (nthlumi > nextlumi_) {
        FillPlots(lumiSeg,lastlumi_,nextlumi_,nthlumi);
        nextlumi_ = nthlumi;
      edm::LogInfo("BeamSpotProblemMonitor") << "beginLuminosityBlock:: Next Lumi to Fit: " << nextlumi_ << endl;
    }
  }
  else{
    if (processed_) FillPlots(lumiSeg,lastlumi_,nextlumi_,nthlumi);
    nextlumi_ = nthlumi;
    edm::LogInfo("BeamSpotProblemMonitor") << " beginLuminosityBlock:: Next Lumi to Fit: " << nextlumi_ << endl;
  }

  if (processed_) processed_ = false;
  edm::LogInfo("BeamSpotProblemMonitor") << " beginLuminosityBlock::  Begin of Lumi: " << nthlumi << endl;
}




// ----------------------------------------------------------
void BeamSpotProblemMonitor::analyze(const Event& iEvent,
			  const EventSetup& iSetup ) {
  const int nthlumi = iEvent.luminosityBlock();


  if (onlineMode_ && (nthlumi < nextlumi_)) {
    edm::LogInfo("BeamSpotProblemMonitor") << "analyze::  Spilt event from previous lumi section!" << std::endl;
    return;
  }
  if (onlineMode_ && (nthlumi > nextlumi_)) {
    edm::LogInfo("BeamSpotProblemMonitor") << "analyze::  Spilt event from next lumi section!!!" << std::endl;
    return;
  }

    BeamSpotStatus_ = 0.;


    // Checking TK status
    Handle<DcsStatusCollection> dcsStatus;
    iEvent.getByLabel("scalersRawToDigi", dcsStatus);
    for (int i=0;i<6;i++) dcsTk[i]=true;
    for (DcsStatusCollection::const_iterator dcsStatusItr = dcsStatus->begin(); 
         dcsStatusItr != dcsStatus->end(); ++dcsStatusItr) {
      if (!dcsStatusItr->ready(DcsStatus::BPIX))   dcsTk[0]=false;
      if (!dcsStatusItr->ready(DcsStatus::FPIX))   dcsTk[1]=false;
      if (!dcsStatusItr->ready(DcsStatus::TIBTID)) dcsTk[2]=false;
      if (!dcsStatusItr->ready(DcsStatus::TOB))    dcsTk[3]=false;
      if (!dcsStatusItr->ready(DcsStatus::TECp))   dcsTk[4]=false;
      if (!dcsStatusItr->ready(DcsStatus::TECm))   dcsTk[5]=false;
    }  

     bool AllTkOn = true;
        for (int i=0; i<5; i++) 
            {
              if (!dcsTk[i]) {
                              AllTkOn = false;
                       
                             }
            }


     //If tracker is ON and collision is going on then must be few track ther
     edm::Handle<reco::TrackCollection> TrackCollection;
     iEvent.getByLabel(trkSrc_, TrackCollection);
     const reco::TrackCollection *tracks = TrackCollection.product();
     for ( reco::TrackCollection::const_iterator track = tracks->begin();track != tracks->end();++track ) 
      {
         if(track->pt() > 1.0)Ntracks_++;
          if(Ntracks_> 200) break;
      }



  // get scalar collection and BeamSpot
  Handle<BeamSpotOnlineCollection> handleScaler;
  iEvent.getByLabel( scalertag_, handleScaler);
     
   // beam spot scalar object
   BeamSpotOnline spotOnline;
     
   bool fallBackToDB=false;
        ALARM_ON_  = false;

 
   if (handleScaler->size()!=0)
     {
      spotOnline = * ( handleScaler->begin() );

    // check if we have a valid beam spot fit result from online DQM thrugh scalars
    if ( spotOnline.x() == 0. &&
         spotOnline.y() == 0. &&
         spotOnline.z() == 0. &&
         spotOnline.width_x() == 0. &&
         spotOnline.width_y() == 0. ) 
      { 
         fallBackToDB=true;
        } 
     }


   //For testing set it false for every LSs
   if(doTest_)fallBackToDB= true;
 
    //based on last event of this lumi only as it overwrite it
    if(AllTkOn && fallBackToDB){BeamSpotStatus_ = -1.;}    //i.e,from DB
    if(AllTkOn && (!fallBackToDB)){BeamSpotStatus_ = 1.;}  //i.e,from online DQM


    //when collision at least few tracks should be there otherwise it give false ALARM  
    if(AllTkOn && Ntracks_ < nCosmicTrk_)BeamSpotStatus_ = 0.;
      

    dbe_->setCurrentFolder(monitorName_+"FitFromScalars/");


  processed_ = true;

}

//--------------------------------------------------------
void BeamSpotProblemMonitor::FillPlots(const LuminosityBlock& lumiSeg,int &lastlumi,int &nextlumi,int &nthlumi){

  if (onlineMode_ && (nthlumi <= nextlumi)) return;

  int currentlumi = nextlumi;
      lastlumi = currentlumi;

    //Chcek status and if lumi are in succession when fall to DB
   if(BeamSpotStatus_== -1.  && (lastlumi+1) == nthlumi)
                             {BeamSpotFromDB_++;
                                 }
                                 else{
                                        BeamSpotFromDB_=0;} //if not in succesion or status is ok then set zero


   if(BeamSpotFromDB_ >= alarmONThreshold_ ){ ALARM_ON_ =true; //set the audio alarm true after N successive LSs
                                           }

   if(BeamSpotFromDB_ > alarmOFFThreshold_ ){ ALARM_ON_ =false; //set the audio alarm true after 10 successive LSs
                                               BeamSpotFromDB_=0; //reset it for new incident
                                             }



 if (onlineMode_) 
 { // filling LS gap For status plot
    const int countLS_bs = hs["BeamSpotStatus_lumi"]->getTH1()->GetEntries();
    int LSgap_bs = currentlumi/fitNLumi_ - countLS_bs;
    if (currentlumi%fitNLumi_ == 0)LSgap_bs--;


    // filling previous fits if LS gap ever exists
    for (int ig = 0; ig < LSgap_bs; ig++) {
      hs["BeamSpotStatus_lumi"]->ShiftFillLast( 0., 0., fitNLumi_ );//x0 , x0err, fitNLumi_;  see DQMCore....
     }

     hs["BeamSpotStatus_lumi"]->ShiftFillLast( BeamSpotStatus_, 0. , fitNLumi_ ); //BeamSpotStatus_ =>0. (no collision, no tracks); =>1 (OK from scaler), =>-1 (No scalar results) 
     hs["BeamSpotStatus_lumi_all"]->setBinContent( currentlumi, BeamSpotStatus_);


 }//onlineMode_
 else { 
      hs["BeamSpotStatus_lumi"]->ShiftFillLast( 0., 0., fitNLumi_ );
     }

      //Reset it here for next lumi
      BeamSpotError->Reset();
      if(ALARM_ON_)BeamSpotError->Fill(BeamSpotFromDB_);


    //Get quality report
     MonitorElement* myQReport = dbe_->get(monitorName_+"FitFromScalars/BeamSpotError");

     const QReport * BeamSpotQReport = myQReport->getQReport("BeamSpotOnlineTest");  

    if(BeamSpotQReport){  
                         float qtresult = BeamSpotQReport->getQTresult();
                         int qtstatus   = BeamSpotQReport->getStatus() ; // get QT status value (see table below)
                         std::string qtmessage = BeamSpotQReport->getMessage() ; // get the whole QT result message
                      }


   Ntracks_= 0;

}

//--------------------------------------------------------
void BeamSpotProblemMonitor::endLuminosityBlock(const LuminosityBlock& lumiSeg,
				     const EventSetup& iSetup) {
  int nthlumi = lumiSeg.id().luminosityBlock();
  edm::LogInfo("BeamSpotProblemMonitor") << "endLuminosityBlock:: Lumi of the last event before endLuminosityBlock: " << nthlumi << endl;

  if (onlineMode_ && nthlumi < nextlumi_) return;

}
//-------------------------------------------------------

void BeamSpotProblemMonitor::endRun(const Run& r, const EventSetup& context){

if(debug_)edm::LogInfo("BeamSpotProblemMonitor") << "endRun:: Clearing all the Maps "<<endl;
     //Reset it end of job
     BeamSpotError->Reset();
}

//--------------------------------------------------------
void BeamSpotProblemMonitor::endJob(const LuminosityBlock& lumiSeg,
			 const EventSetup& iSetup){
  if (!onlineMode_) endLuminosityBlock(lumiSeg, iSetup);
     //Reset it end of job
     BeamSpotError->Reset();
}

DEFINE_FWK_MODULE(BeamSpotProblemMonitor);
