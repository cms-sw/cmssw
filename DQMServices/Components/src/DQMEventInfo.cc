/*
 * \file DQMEventInfo.cc
 * \author M. Zanetti - CERN PH
 * Last Update:
 * $Date: 2010/06/01 18:06:24 $
 * $Revision: 1.28 $
 * $Author: dellaric $
 *
 */
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMEventInfo.h"
#include <TSystem.h>

#include <stdio.h>
#include <sstream>
#include <math.h>


static inline double stampToReal(edm::Timestamp time)
{ return (time.value() >> 32) + 1e-6*(time.value() & 0xffffffff); }

static inline double stampToReal(const timeval &time)
{ return time.tv_sec + 1e-6*time.tv_usec; }


DQMEventInfo::DQMEventInfo(const edm::ParameterSet& ps){
  
  struct timeval now;
  gettimeofday(&now, 0);

  parameters_ = ps;
  pEvent_ = 0;
  evtRateCount_ = 0;
  lastAvgTime_ = currentTime_ = stampToReal(now);

  // read config parms  
  std::string folder = parameters_.getUntrackedParameter<std::string>("eventInfoFolder", "EventInfo") ;
  std::string subsystemname = parameters_.getUntrackedParameter<std::string>("subSystemFolder", "YourSubsystem") ;
  
  eventInfoFolder_ = subsystemname + "/" +  folder ;
  evtRateWindow_ = parameters_.getUntrackedParameter<double>("eventRateWindow", 0.5);
  if(evtRateWindow_<=0.15) evtRateWindow_=0.15;

  nameProcess_ = parameters_.getUntrackedParameter<std::string>( "processName", "HLT" );

  // 
  dbe_ = edm::Service<DQMStore>().operator->();

  dbe_->setCurrentFolder(eventInfoFolder_) ;

  //Event specific contents
  runId_     = dbe_->bookInt("iRun");
  runId_->Fill(-1);
  lumisecId_ = dbe_->bookInt("iLumiSection");
  lumisecId_->Fill(-1);
  eventId_   = dbe_->bookInt("iEvent");
  eventId_->Fill(-1);
  eventTimeStamp_ = dbe_->bookFloat("eventTimeStamp");
  
  isCollisionsRun_ = dbe_->bookInt("isCollisionsRun");
  isCollisionsRun_->Fill(0);
  
  dbe_->setCurrentFolder(eventInfoFolder_) ;
  //Process specific contents
  processTimeStamp_ = dbe_->bookFloat("processTimeStamp");
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
  processStartTimeStamp_->Fill(currentTime_);
  runStartTimeStamp_ = dbe_->bookFloat("runStartTimeStamp");
  hostName_= dbe_->bookString("hostName",gSystem->HostName());
  processName_= dbe_->bookString("processName",subsystemname);
  workingDir_= dbe_->bookString("workingDir",gSystem->pwd());
  cmsswVer_= dbe_->bookString("CMSSW_Version",edm::getReleaseVersion());
 
  // Folder to be populated by sub-systems' code
  std::string subfolder = eventInfoFolder_ + "/reportSummaryContents" ;
  dbe_->setCurrentFolder(subfolder);

}

DQMEventInfo::~DQMEventInfo(){
}

void DQMEventInfo::beginRun(const edm::Run& r, const edm::EventSetup &c ) {
    
  runId_->Fill(r.id().run());
  runStartTimeStamp_->Fill(stampToReal(r.beginTime()));

  std::string hltKey = "";
  HLTConfigProvider hltConfig;
  bool changed( true );
  if ( ! hltConfig.init( r, c, nameProcess_, changed) ) 
  {
    // std::cout << "errorHltConfigExtraction" << std::endl;
    hltKey = "error extraction" ;
  } 
  else if ( hltConfig.size() <= 0 ) 
  {
   // std::cout << "hltConfig" << std::endl;
    hltKey = "error key of length 0" ;
  } 
  else 
  {
    std::cout << "HLT key (run)  : " << hltConfig.tableName() << std::endl;
    hltKey =  hltConfig.tableName() ;
  }

  dbe_->setCurrentFolder(eventInfoFolder_) ;
  hltKey_= dbe_->bookString("hltKey",hltKey);
  
} 

void DQMEventInfo::beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c) {

  lumisecId_->Fill(l.id().luminosityBlock());

}

void DQMEventInfo::analyze(const edm::Event& e, const edm::EventSetup& c){
 
  eventId_->Fill(int64_t(e.id().event()));
  eventTimeStamp_->Fill(stampToReal(e.time()));

  pEvent_++;
  evtRateCount_++;
  processEvents_->Fill(pEvent_);

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
  


  // get beam mode from GT
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvm_handle;
  if (gtEvm_handle.isValid()) 
    e.getByLabel("gtEvmDigis", gtEvm_handle);
  else
    return;
    
  L1GlobalTriggerEvmReadoutRecord const* gtevm = gtEvm_handle.product();

  // not needed:   L1GtfeWord gtfeEvmWord;
  L1GtfeExtWord gtfeEvmExtWord;
  if (gtevm)
  {
     // not needed:   gtfeEvmWord = gtevm->gtfeWord();
     gtfeEvmExtWord = gtevm->gtfeWord();
  }
  else
  {
    // std::cout << " gtfeEvmWord inaccessible" ;
    return;
  }
   
  int beamMode = gtfeEvmExtWord.beamMode();
  // std::cout << "beamMode: " << beamMode << std::endl;
  if (beamMode == 11) 
    isCollisionsRun_->Fill(1);

  return;
}
