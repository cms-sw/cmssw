/*
 * \file DQMFileSaver.cc
 * 
 * $Date: 2008/02/21 03:26:49 $
 * $Revision: 1.8 $
 * $Author: lat $
 * \author A. Meyer, DESY
 *
 */

#include "DQMFileSaver.h"

// Framework


#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace std;

//--------------------------------------------------------
DQMFileSaver::DQMFileSaver():
irun_(0), ilumisec_(0), ievent_(0), itime_(0),
counterEvt_(0), counterLS_(0)
{}

//--------------------------------------------------------
DQMFileSaver::DQMFileSaver(const ParameterSet& ps):
irun_(0), ilumisec_(0), ievent_(0), itime_(0),
counterEvt_(0), counterLS_(0)
{
  parameters_ = ps;
  initialize();
}

//--------------------------------------------------------
void DQMFileSaver::initialize(){  
  
  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();
    
  // set parameters   
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  edm::LogVerbatim ("DQMFileSaver") << "===> save every " << prescaleEvt_ << " event(s)"<< endl;

  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
  edm::LogVerbatim ("DQMFileSaver") << "===> save every " << prescaleLS_ << " lumi section(s)"<< endl;

  prescaleTime_ = parameters_.getUntrackedParameter<int>("prescaleTime", -1);
  edm::LogVerbatim ("DQMFileSaver") << "===> save every " << prescaleTime_ << " minutes(s)"<< endl;
  
  saveAtRunEnd_ = parameters_.getUntrackedParameter<bool>("saveAtRunEnd",true);
  (saveAtRunEnd_)? edm::LogVerbatim ("DQMFileSaver") << "===> save at run end " << endl :
                   edm::LogVerbatim ("DQMFileSaver") << "===> NO save at run end " << endl ; 
  
  saveAtJobEnd_ = parameters_.getUntrackedParameter<bool>("saveAtJobEnd",false);
  (saveAtJobEnd_)? edm::LogVerbatim ("DQMFileSaver") << "===> save at Job end " << endl :
                   edm::LogVerbatim ("DQMFileSaver") << "===> NO save at Job end " << endl ; 
  
  // Base filename for the contents of this job
  fileName_ = "DQM_"+parameters_.getUntrackedParameter<string>("fileName","YourSubsystemName");
  edm::LogVerbatim ("DQMFileSaver") << "===>DQM Output file name = " << fileName_ << endl;
  // Give different filename in case it is from playback
  isPlayback_ = parameters_.getUntrackedParameter<bool>("isPlayback",false);
  edm::LogVerbatim ("DQMFileSaver") << "===>DQM Output from Playback = " << ( isPlayback_ ? "yes" : "no" ) << endl;
  if ( isPlayback_ ) fileName_ = "Playback_"+fileName_;
  // dirname for the file to be written
  dirName_ = parameters_.getUntrackedParameter<string>("dirName",".");
  if (dirName_ == "" ) dirName_ == "." ;
  edm::LogVerbatim ("DQMFileSaver") << "===>DQM Output dir name = " << dirName_ << endl;
    
  gettimeofday(&psTime_.startTV,NULL);
  /// get time in milliseconds, convert to minutes
  psTime_.startTime = (psTime_.startTV.tv_sec*1000.0+psTime_.startTV.tv_usec/1000.0);
  psTime_.startTime /= (1000.0*60.0);
  psTime_.elapsedTime=0;
  psTime_.updateTime=0;

}

//--------------------------------------------------------
DQMFileSaver::~DQMFileSaver(){

  edm::LogVerbatim ("DQMFileSaver")<<"DQMFileSaver::destructor"<<endl;

}

//--------------------------------------------------------
void DQMFileSaver::beginJob(const EventSetup& c){
  
  edm::LogVerbatim ("DQMFileSaver")<<"DQMFileSaver::begin job"<<endl;

  counterEvt_=0;
  counterLS_=0;
  
}

//--------------------------------------------------------
void DQMFileSaver::analyze(const Event& e, const EventSetup& c){
 
  counterEvt_++;
  if (counterEvt_==1) edm::LogVerbatim ("DQMFileSaver")<<"DQMFileSaver::analyze"<<endl;

  // environment datamembers
  irun_     = e.id().run();
  ilumisec_ = e.luminosityBlock();
  ievent_   = e.id().event();
  itime_    = e.time().value();

  // cout << "DQMFileSaver: evts: "<< counterEvt_ << ", run: " << irun_ << ", LS: " << ilumisec_ << ", evt: " << ievent_ << ", time: " << itime_ << endl; 
  
  // save every n events
  if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ == 0 ) {
    // take event and run number from Event setup
    char run[10]; sprintf(run,"%09d", irun_);
    char evt[10]; sprintf(evt,"%08d", ievent_);
    string outFile = dirName_+"/"+fileName_+"_R"+run+"_E"+evt+".root";
    dbe_->save(outFile);
  }

  // save every n minutes  
  if (prescaleTime_>0 ) {
    //get elapsed time in minutes...
    gettimeofday(&psTime_.updateTV,NULL);
    double currTime =(psTime_.updateTV.tv_sec*1000.0+psTime_.updateTV.tv_usec/1000.0); //in milliseconds
    currTime /= (1000.0*60.0); //in minutes
    psTime_.elapsedTime = currTime - psTime_.startTime;
    
    float counterTime = psTime_.elapsedTime - psTime_.updateTime;

    // cout << " time " << counterTime << " " << psTime_.elapsedTime << " " 
    //                          << psTime_.updateTime << " " 
    //			        << psTime_.startTime <<  endl;
     if(counterTime > prescaleTime_){
       psTime_.updateTime = psTime_.elapsedTime;
       
       char run[10]; sprintf(run,"%09d", irun_);
       char time[10]; sprintf(time,"%08d", (int)psTime_.elapsedTime);
       string outFile = dirName_ +"/"+fileName_+"_R"+run+"_T"+time+".root";
       dbe_->save(outFile);
     }
  }

}

//--------------------------------------------------------
void DQMFileSaver::beginRun(const Run& r, const EventSetup& c){
}

//--------------------------------------------------------
void DQMFileSaver::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& c){
   counterLS_++;
   if (counterLS_==1) edm::LogVerbatim ("DQMFileSaver") <<"DQMFileSaver::beginLuminosityBlock"<<endl;
}

//--------------------------------------------------------
void DQMFileSaver::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& c){
   if (counterLS_==1) edm::LogVerbatim ("DQMFileSaver") <<"DQMFileSaver::endLuminosityBlock"<<endl;
   if (prescaleLS_<0) return ;
   if (counterLS_%prescaleLS_ != 0 ) return;
 
   // add lumisection number (get from event)
   char run[10]; sprintf(run,"%09d", irun_);
   char lumisec[10]; sprintf(lumisec,"%06d", ilumisec_);
   string outFile = dirName_+"/"+fileName_+"_R"+run+"_L"+lumisec+".root";
   dbe_->save(outFile);

}

//--------------------------------------------------------
void DQMFileSaver::endRun(const Run& r, const EventSetup& c){
   if (saveAtRunEnd_) {
     char run[10];
     if(irun_>0) sprintf(run,"%09d", irun_);
     else sprintf(run,"%09d", 0);
     string outFile = dirName_+"/"+fileName_+"_R"+run+".root";
     dbe_->save(outFile,"",irun_);
   }
}

//--------------------------------------------------------
void DQMFileSaver::endJob() { 
   if (saveAtJobEnd_) {
     string outFile = dirName_+"/"+fileName_+".root";
     dbe_->save(outFile);
   }   
}
