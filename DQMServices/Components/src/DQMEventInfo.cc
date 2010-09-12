/*
 * \file DQMEventInfo.cc
 * \author M. Zanetti - CERN PH
 * Last Update:
 * $Date: 2010/06/01 18:06:24 $
 * $Revision: 1.28 $
 * $Author: dellaric $
 *
 */

#include "DQMEventInfo.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include <TSystem.h>

// Framework


#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

static inline double stampToReal(edm::Timestamp time)
{ return (time.value() >> 32) + 1e-6*(time.value() & 0xffffffff); }

static inline double stampToReal(const timeval &time)
{ return time.tv_sec + 1e-6*time.tv_usec; }

DQMEventInfo::DQMEventInfo(const ParameterSet& ps){
  
  struct timeval now;
  gettimeofday(&now, 0);

  parameters_ = ps;
  pEvent_ = 0;
  evtRateCount_ = 0;
//  gettimeofday(&currentTime_,NULL);
//  lastAvgTime_ = currentTime_;
  lastAvgTime_ = currentTime_ = stampToReal(now);
  
  dbe_ = edm::Service<DQMStore>().operator->();

  string eventinfofolder = parameters_.getUntrackedParameter<string>("eventInfoFolder", "EventInfo") ;
  string subsystemname = parameters_.getUntrackedParameter<string>("subSystemFolder", "YourSubsystem") ;
  string currentfolder = subsystemname + "/" +  eventinfofolder ;

  evtRateWindow_ = parameters_.getUntrackedParameter<double>("eventRateWindow", 0.5);
  if(evtRateWindow_<=0.15) evtRateWindow_=0.15;

  dbe_->setCurrentFolder(currentfolder) ;

  //Event specific contents
  runId_     = dbe_->bookInt("iRun");
  runId_->Fill(-1);
  lumisecId_ = dbe_->bookInt("iLumiSection");
  lumisecId_->Fill(-1);
  eventId_   = dbe_->bookInt("iEvent");
  eventId_->Fill(-1);
  eventTimeStamp_ = dbe_->bookFloat("eventTimeStamp");
  
  dbe_->setCurrentFolder(currentfolder) ;
  //Process specific contents
  processTimeStamp_ = dbe_->bookFloat("processTimeStamp");
//  processTimeStamp_->Fill(getUTCtime(&currentTime_));
  processTimeStamp_->Fill(currentTime_);
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
//  processStartTimeStamp_->Fill(getUTCtime(&currentTime_));
  processStartTimeStamp_->Fill(currentTime_);
  runStartTimeStamp_ = dbe_->bookFloat("runStartTimeStamp");
  hostName_= dbe_->bookString("hostName",gSystem->HostName());
  processName_= dbe_->bookString("processName",subsystemname);
  workingDir_= dbe_->bookString("workingDir",gSystem->pwd());
  cmsswVer_= dbe_->bookString("CMSSW_Version",edm::getReleaseVersion());
//  dqmPatch_= dbe_->bookString("DQM_Patch",dbe_->getDQMPatchVersion());
 
  // Folder to be populated by sub-systems' code
  string subfolder = currentfolder + "/reportSummaryContents" ;
  dbe_->setCurrentFolder(subfolder);

}

DQMEventInfo::~DQMEventInfo(){
}

void DQMEventInfo::beginRun(const edm::Run& r, const edm::EventSetup &c ) {
    
  runId_->Fill(r.id().run());

//   const edm::Timestamp time = r.beginTime();
// 
//   float sec = time.value() >> 32; 
//   float usec = 0xFFFFFFFF & time.value() ; 
// 
//   // cout << " begin Run " << r.run() << " " << time.value() << endl;
//   // cout << setprecision(16) << getUTCtime(&currentTime_) << endl;
//   // cout << sec+usec/1000000. << endl;
// 
//   runStartTimeStamp_->Fill(sec+usec/1000000.);
  runStartTimeStamp_->Fill(stampToReal(r.beginTime()));
  
} 

void DQMEventInfo::beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c) {

  lumisecId_->Fill(l.id().luminosityBlock());

}

void DQMEventInfo::analyze(const Event& e, const EventSetup& c){
 
  eventId_->Fill(int64_t(e.id().event()));
//  eventTimeStamp_->Fill(e.time().value()/(double)0xffffffff);
  eventTimeStamp_->Fill(stampToReal(e.time()));

  pEvent_++;
  evtRateCount_++;
  processEvents_->Fill(pEvent_);

//  lastUpdateTime_=currentTime_;
//  gettimeofday(&currentTime_,NULL);  
//  processTimeStamp_->Fill(getUTCtime(&currentTime_));
//  processLatency_->Fill(getUTCtime(&lastUpdateTime_,&currentTime_));
//
//  float time = getUTCtime(&lastAvgTime_,&currentTime_);
//  if(time>=(evtRateWindow_*60.0)){
//    processEventRate_->Fill((float)evtRateCount_/time);
  struct timeval now;
  gettimeofday(&now, 0);
  lastUpdateTime_ = currentTime_;
  currentTime_ = stampToReal(now);

  processTimeStamp_->Fill(currentTime_);
  processLatency_->Fill(currentTime_ - lastUpdateTime_);

  double delta = currentTime_ - lastAvgTime_;
  if (delta >= (evtRateWindow_*60.0))
  {
    processEventRate_->Fill(evtRateCount_/delta);
    evtRateCount_ = 0;
    lastAvgTime_ = currentTime_;    
  }

  return;
}
//
//double DQMEventInfo::getUTCtime(timeval* a, timeval* b){
//  double deltaT=(*a).tv_sec*1000.0+(*a).tv_usec/1000.0;
//  if(b!=NULL) deltaT=(*b).tv_sec*1000.0+(*b).tv_usec/1000.0 - deltaT;
//  return deltaT/1000.0;
//}
