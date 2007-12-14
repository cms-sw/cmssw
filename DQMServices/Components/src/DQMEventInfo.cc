/*
 * \file DQMEventInfo.cc
 * \author M. Zanetti - CERN PH
 * Last Update:
 * $Date: 2007/11/21 00:19:12 $
 * $Revision: 1.8 $
 * $Author: wfisher $
 *
 */

#include "DQMEventInfo.h"
#include <TSystem.h>

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

using namespace edm;
using namespace std;

DQMEventInfo::DQMEventInfo(const ParameterSet& ps){
  
  parameters_ = ps;
  pEvent_ = 0;
  gettimeofday(&currentTime_,NULL);
  
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();

  string eventinfofolder = parameters_.getUntrackedParameter<string>("eventInfoFolder", "EventInfo") ;
  string subsystemname = parameters_.getUntrackedParameter<string>("subSystemFolder", "YourSubsystem") ;
  string currentfolder = subsystemname + "/" +  eventinfofolder ;
  cout << "currentfolder " << currentfolder << endl;

  dbe_->setCurrentFolder(currentfolder) ;

  //Event specific contents
  runId_     = dbe_->bookInt("iRun");
  lumisecId_ = dbe_->bookInt("iLumiSection");
  eventId_   = dbe_->bookInt("iEvent");
  eventTimeStamp_ = dbe_->bookFloat("eventTimeStamp");
  errSummary_ = dbe_->bookFloat("errorSummary");
  errSummary_->Fill(-1);

  //Process specific contents
  processTimeStamp_ = dbe_->bookFloat("processTimeStamp");
  processTimeStamp_->Fill(getUTCtime(&currentTime_));
  processLatency_ = dbe_->bookFloat("processLatency");
  processTimeStamp_->Fill(-1);
  processEvents_ = dbe_->bookInt("processedEvents");
  processEvents_->Fill(pEvent_);
  nUpdates_= dbe_->bookInt("processUpdates");
  nUpdates_->Fill(-1);
  

  //Static Contents
  processId_= dbe_->bookInt("processID"); 
  processId_->Fill(gSystem->GetPid());
  hostName_= dbe_->bookString("hostName",gSystem->HostName());
  processName_= dbe_->bookString("processName",subsystemname);
  workingDir_= dbe_->bookString("workingDir",gSystem->pwd());
  cmsswVer_= dbe_->bookString("CMSSW_Version",edm::getReleaseVersion());
  dqmPatch_= dbe_->bookString("DQM_Patch",dbe_->getDQMPatchVersion());
    
}

DQMEventInfo::~DQMEventInfo(){

  cout<<"[DQMEventInfo]: destructor"<<endl;

}

void DQMEventInfo::analyze(const Event& e, const EventSetup& c){

  runId_->Fill(e.id().run());
  lumisecId_->Fill(e.luminosityBlock());
  eventId_->Fill(e.id().event());
  eventTimeStamp_->Fill(e.time().value());

  pEvent_++;
  processEvents_->Fill(pEvent_);

  lastUpdateTime_=currentTime_;
  gettimeofday(&currentTime_,NULL);  
  processTimeStamp_->Fill(getUTCtime(&currentTime_));
  processLatency_->Fill(getUTCtime(&lastUpdateTime_,&currentTime_));

}

double DQMEventInfo::getUTCtime(timeval* a, timeval* b){
  double deltaT=(*a).tv_sec*1000.0+(*a).tv_usec/1000.0;
  if(b!=NULL) deltaT=(*b).tv_sec*1000.0+(*b).tv_usec/1000.0 - deltaT;
  return deltaT/1000.0;
}
