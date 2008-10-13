/*
 * \file DQMEventInfo.cc
 * \author M. Zanetti - CERN PH
 * Last Update:
 * $Date: 2008/07/06 17:32:30 $
 * $Revision: 1.21 $
 * $Author: ameyer $
 *
 */

#include "DQMServices/XdaqCollector/interface/XmasToDQMEventInfo.h"
#include <TSystem.h>

// Framework


#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

XmasToDQMEventInfo::XmasToDQMEventInfo(const ParameterSet& ps){

 cout << "Constructor of XmasToDQMEventInfo called...." << endl;
  
  parameters_ = ps;
  pEvent_ = 0;
  evtRateCount_ = 0;
  gettimeofday(&currentTime_,NULL);
  lastAvgTime_ = currentTime_;
  
  dbe_ = edm::Service<DQMStore>().operator->();

  string eventinfofolder = parameters_.getUntrackedParameter<string>("eventInfoFolder", "EventInfo") ;
  string subsystemname = parameters_.getUntrackedParameter<string>("subSystemFolder", "YourSubsystem") ;
  string currentfolder = subsystemname + "/" +  eventinfofolder ;
  cout << "currentfolder " << currentfolder << endl;

  evtRateWindow_ = parameters_.getUntrackedParameter<double>("eventRateWindow", 0.5);
  if(evtRateWindow_<=0.15) evtRateWindow_=0.15;
  cout << "Event Rate averaged over " << evtRateWindow_ << " minutes" << endl;

  dbe_->setCurrentFolder(currentfolder) ;

  //Event specific contents
  runId_     = dbe_->bookInt("iRun");
  lumisecId_ = dbe_->bookInt("iLumiSection");
  eventId_   = dbe_->bookInt("iEvent");
  eventTimeStamp_ = dbe_->bookFloat("eventTimeStamp");
  
  dbe_->setCurrentFolder(currentfolder) ;
  //Process specific contents
  processTimeStamp_ = dbe_->bookFloat("processTimeStamp");
  processTimeStamp_->Fill(getUTCtime(&currentTime_));
  processLatency_ = dbe_->bookFloat("processLatency");
  processTimeStamp_->Fill(-1);
  processEvents_ = dbe_->bookInt("processedEvents");
  processEvents_->Fill(pEvent_);
  processEventRate_ = dbe_->bookFloat("processEventRate");
  processEventRate_->Fill(-1); 
  nUpdates_= dbe_->bookInt("processUpdates");
  nUpdates_->Fill(-1);

  //Static Contents
  processId_= dbe_->bookInt("processID"); 
  processId_->Fill(gSystem->GetPid());
  processStartTimeStamp_ = dbe_->bookFloat("processStartTimeStamp");
  processStartTimeStamp_->Fill(getUTCtime(&currentTime_));
  runStartTimeStamp_ = dbe_->bookFloat("runStartTimeStamp");
  hostName_= dbe_->bookString("hostName",gSystem->HostName());
  processName_= dbe_->bookString("processName",subsystemname);
  workingDir_= dbe_->bookString("workingDir",gSystem->pwd());
  cmsswVer_= dbe_->bookString("CMSSW_Version",edm::getReleaseVersion());
  dqmPatch_= dbe_->bookString("DQM_Patch",dbe_->getDQMPatchVersion());
 
  // Folder to be populated by sub-systems' code
  string subfolder = currentfolder + "/reportSummaryContents" ;
  dbe_->setCurrentFolder(subfolder);

}

XmasToDQMEventInfo::~XmasToDQMEventInfo(){

  cout<<"[XmasToDQMEventInfo]: destructor"<<endl;

}

void XmasToDQMEventInfo::beginRun(const edm::Run& r, const edm::EventSetup &c ) {
  
  const edm::Timestamp time = r.beginTime();

  float sec = time.value() >> 32; 
  float usec = 0xFFFFFFFF & time.value() ; 

  // cout << " begin Run " << r.run() << " " << time.value() << endl;
  // cout << setprecision(16) << getUTCtime(&currentTime_) << endl;
  // cout << sec+usec/1000000. << endl;

  runStartTimeStamp_->Fill(sec+usec/1000000.);
  
} 

void XmasToDQMEventInfo::analyze(const Event& e, const EventSetup& c){
 
  static int icalls = 1;
  cout << "XmasToDQMEventInfo::analyze call = " << icalls << std::endl;
  
  //runId_->Fill(e.id().run());
  std::string runNumber;
  
  xmas2dqm::wse::ToDqm::instance()->BSem_.take();
  runNumber = xmas2dqm::wse::ToDqm::instance()->runNumber_;
  xmas2dqm::wse::ToDqm::instance()->BSem_.give();
  
  std::istringstream istrfloat(runNumber);
  float runNumber_value;
  istrfloat >> runNumber_value;
  runId_->Fill(runNumber_value);
  
  lumisecId_->Fill(e.luminosityBlock());
  eventId_->Fill(e.id().event());
  eventTimeStamp_->Fill(e.time().value()/(double)0xffffffff);

  pEvent_++;
  evtRateCount_++;
  processEvents_->Fill(pEvent_);

  lastUpdateTime_=currentTime_;
  gettimeofday(&currentTime_,NULL);  
  processTimeStamp_->Fill(getUTCtime(&currentTime_));
  processLatency_->Fill(getUTCtime(&lastUpdateTime_,&currentTime_));

  float time = getUTCtime(&lastAvgTime_,&currentTime_);
  if(time>=(evtRateWindow_*60.0)){
    processEventRate_->Fill((float)evtRateCount_/time);
    evtRateCount_ = 0;
    lastAvgTime_ = currentTime_;    
  }

  icalls++;
  return;
}

double XmasToDQMEventInfo::getUTCtime(timeval* a, timeval* b){
  double deltaT=(*a).tv_sec*1000.0+(*a).tv_usec/1000.0;
  if(b!=NULL) deltaT=(*b).tv_sec*1000.0+(*b).tv_usec/1000.0 - deltaT;
  return deltaT/1000.0;
}
