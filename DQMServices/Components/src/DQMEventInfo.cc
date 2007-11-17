/*
 * \file DQMEventInfo.cc
 * \author M. Zanetti - CERN PH
 * Last Update:
 * $Date: 2007/11/15 23:09:28 $
 * $Revision: 1.5 $
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
  timer_.start();

  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe_->setVerbose(1);

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


  //Process specific contents
  processTimeStamp_ = dbe_->bookFloat("processTimeStamp");
  processTimeStamp_->Fill(timer_.realTime());
  processEvents_ = dbe_->bookInt("processedEvents");
  processEvents_->Fill(pEvent_);
  nUpdates_= dbe_->bookInt("processUpdates");
  nUpdates_->Fill(0);
  

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
  processTimeStamp_->Fill(timer_.realTime());
  //alternatively can use timer_.cpuTime() for system clock timestamp

}
