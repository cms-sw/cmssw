/*
 * \file DQMAnalyzer.cc
 * 
 * $Date: 2007/10/12 21:19:28 $
 * $Revision: 1.3 $
 * \author M. Zanetti - CERN PH
 *
 */

#include "DQMServices/Components/interface/DQMAnalyzer.h"

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace std;

//--------------------------------------------------------
DQMAnalyzer::DQMAnalyzer():
irun_(0), ilumisec_(0), ievent_(0), itime_(0),
actonLS_(false),debug_(false),
nevt_(0), nlumisecs_(0),
saved_(false)
{}

//--------------------------------------------------------
DQMAnalyzer::DQMAnalyzer(const ParameterSet& ps):
irun_(0), ilumisec_(0), ievent_(0), itime_(0),
actonLS_(false),debug_(false),
nevt_(0), nlumisecs_(0),
saved_(false)
{
  parameters_ = ps;
  initialize();
}

//--------------------------------------------------------
void DQMAnalyzer::initialize(){  

  debug_ = parameters_.getUntrackedParameter<bool>("debug", false);
  if(debug_) cout << "DQMAnalyzer: constructor...." << endl;
  
  // get back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  if(debug_) cout << "===>DQM DaqMonitorBEInterface " << endl;
  
  // some other environment parms
  if ( parameters_.getUntrackedParameter<bool>("enableMonitorDaemon", true) ) {
    Service<MonitorDaemon> daemon;
    daemon.operator->();
    if(debug_) cout << "===>DQM MonitorDaemon enabled " << endl;
  } 
  else if(debug_) cout<<"DQMAnalyzer: Warning, MonitorDaemon service not enabled"<<endl;
  
  if ( debug_ ) dbe_->setVerbose(1);
  else dbe_->setVerbose(0);

  // set parameters   
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("diagnosticPrescaleEvt", -1);
  cout << "===>DQM event prescale = " << prescaleEvt_ << " event(s)"<< endl;

  prescaleLS_ = parameters_.getUntrackedParameter<int>("diagnosticPrescaleLS", -1);
  cout << "===>DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  if (prescaleLS_>0) actonLS_=true;

  prescaleUpdate_ = parameters_.getUntrackedParameter<int>("diagnosticPrescaleUpdate", -1);
  cout << "===>DQM update prescale = " << prescaleUpdate_ << " update(s)"<< endl;

  prescaleTime_ = parameters_.getUntrackedParameter<int>("diagnosticPrescaleTime", -1);
  cout << "===>DQM time prescale = " << prescaleTime_ << " minute(s)"<< endl;
  
  
  // Base folder for the contents of this job
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","");
  cout << "===>DQM monitor name = " << monitorName_ << endl;
    
  rootFolder_ = "DQMAnalyzer";
  if (monitorName_.size() != 0){
    rootFolder_ = monitorName_ + "Monitor/";
    cout << "===>DQM rootFolder  = " << rootFolder_ << endl;
  }
  
  gettimeofday(&psTime_.startTV,NULL);
  /// get time in milliseconds, convert to minutes
  psTime_.startTime = (psTime_.startTV.tv_sec*1000.0+psTime_.startTV.tv_usec/1000.0);
  psTime_.startTime /= (60.0*1000.0);
  psTime_.elapsedTime=0;
  psTime_.updateTime=0;

}

//--------------------------------------------------------
DQMAnalyzer::~DQMAnalyzer(){

  if (debug_) cout<<"DQMAnalyzer::destructor"<<endl;

}

//--------------------------------------------------------
void DQMAnalyzer::beginJob(const EventSetup& c){
  
  if (debug_) cout<<"DQMAnalyzer::begin job"<<endl;

  nevt_=0;
  nlumisecs_=0;
  
  // book framework ME
  dbe_->setVerbose(1);
  dbe_->setCurrentFolder(parameters_.getUntrackedParameter<string>("eventInfoFolder", "EventInfo/")) ;
  runId_ = dbe_->bookInt("iRun");
  lumisecId_ = dbe_->bookInt("iLumiSection");
  eventId_ = dbe_->bookInt("iEvent");
  timeStamp_ = dbe_->bookFloat("timeStamp");

}

//--------------------------------------------------------
void DQMAnalyzer::analyze(const Event& e, const EventSetup& c){
 
  if (debug_ || nevt_==1) cout<<"DQMAnalyzer::analyze"<<endl;
  
  //get elapsed time in minutes...
  gettimeofday(&psTime_.updateTV,NULL);
  double currTime =(psTime_.updateTV.tv_sec*1000.0+psTime_.updateTV.tv_usec/1000.0); //in milliseconds
  currTime /= (60.0*1000.0); //in minutes
  psTime_.elapsedTime = currTime - psTime_.startTime;

  // set counters and flags
  saved_ = false;
  nevt_++;

  // environment datamembers
  irun_     = e.id().run();
  ilumisec_ = e.luminosityBlock();
  ievent_   = e.id().event();
  itime_    = e.time().value();

  if (debug_) cout << "DQMAnalyzer: evts: "<< nevt_ << ", run: " << irun_ << ", LS: " << ilumisec_ << ", evt: " << ievent_ << ", time: " << itime_ << endl; 
  
  // ME
  if (runId_)     runId_->Fill(irun_);
  if (lumisecId_) lumisecId_->Fill(ilumisec_); 
  if (eventId_)   eventId_->Fill(ievent_);
  if (timeStamp_) timeStamp_->Fill(itime_); 


  if (debug_) prescale();
  
}

//--------------------------------------------------------
void DQMAnalyzer::beginRun(const Run& r, const EventSetup& c){
  cout <<"DQMAnalyzer::begin run "<< r.id().run() << endl;
  cout <<"FIXME reset histos here"<<endl;
}

//--------------------------------------------------------
void DQMAnalyzer::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& c){
   if (debug_ || nlumisecs_==0) cout <<"DQMAnalyzer::beginLuminosityBlock"<<endl;
   nlumisecs_++;

   if(actonLS_ && !prescale()){
     // do scheduled tasks...
   }
   cout <<"nlumisecs_: "<<nlumisecs_<<endl;
}

//--------------------------------------------------------
void DQMAnalyzer::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& c){
   if (debug_ || nlumisecs_==1) cout <<"DQMAnalyzer::endLuminosityBlock"<<endl;

   if(actonLS_ && !prescale()){
     // do scheduled tasks...
     save(); 
   }   
}

//--------------------------------------------------------
void DQMAnalyzer::endRun(const Run& r, const EventSetup& c){
  cout <<"DQMAnalyzer::end run "<< r.id().run() << endl;
   save("endRun");
}

//--------------------------------------------------------
void DQMAnalyzer::endJob() { 
   if (debug_) cout <<"DQMAnalyzer::endJob"<<endl;
   save();
}

//--------------------------------------------------------
void DQMAnalyzer::reset() { 
  if (debug_) cout <<"DQMAnalyzer::reset"<<endl;
}

//--------------------------------------------------------
bool DQMAnalyzer::prescale(){
  ///Return true if this event should be skipped according to the prescale condition...
  ///    Accommodate a logical "OR" of the possible tests
  if (debug_) cout <<"DQMAnalyzer::prescale"<<endl;
  
  //First determine if we care...
  bool evtPS =    prescaleEvt_>0;
  bool lsPS =     prescaleLS_>0;
  bool timePS =   prescaleTime_>0;
  bool updatePS = prescaleUpdate_>0;

  // If no prescales are set, keep the event
  if(!evtPS && !lsPS && !timePS && !updatePS) return false;

  //check each instance
  if(lsPS && (ilumisec_%prescaleLS_)!=0) lsPS = false; //LS veto
  if(evtPS && (ievent_%prescaleEvt_)!=0) evtPS = false; //evt # veto
  if(timePS){
    float time = psTime_.elapsedTime - psTime_.updateTime;
    if(time<prescaleTime_){
      timePS = false;  //timestamp veto
      psTime_.updateTime = psTime_.elapsedTime;
    }
  }
  //  if(prescaleUpdate_>0 && (nupdates_%prescaleUpdate_)==0) updatePS=false; ///need to define what "updates" means
  
  if (debug_) printf("DQMAnalyzer::prescale  evt: %d/%d, ls: %d/%d, time: %f/%d\n",
		     ievent_,evtPS,
		     ilumisec_,lsPS,
		     psTime_.elapsedTime - psTime_.updateTime,timePS);

  // if any criteria wants to keep the event, do so
  if(evtPS || lsPS || timePS) return false; //FIXME updatePS left out for now
  return true;
}

//--------------------------------------------------------
void DQMAnalyzer::save(std::string flag){
  
  if (debug_) cout <<"DQMAnalyzer::save"<<endl;

  bool disable = parameters_.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    cout <<"DQMAnalyzer:  ROOT output disabled"<<endl;
    return;
  }

  if (saved_) return; // save only once per event
  if (debug_) cout <<"DQMAnalyzer::save: saving"<<endl;
  
  std::string name = "DQM_"+monitorName_;
 
  // add runnumber  
  char run[10];
  if(irun_>0) sprintf(run,"%09d", irun_);
  else sprintf(run,"%09d", 0);

  if (flag=="endRun") {
    string outFile = name+"_"+run+".root";
    dbe_->save(outFile);
    saved_=true; // save only once per event
    return;
  }
  
  // add lumisection number  
  char lumisec[10];
  if(ilumisec_>0) sprintf(lumisec,"%06d", ilumisec_);
  else sprintf(lumisec,"%06d", 0);
  
  string outFile = name+"_"+run+"_"+lumisec+".root";
  dbe_->save(outFile);
  saved_=true; // save only once per event
  return;
}

