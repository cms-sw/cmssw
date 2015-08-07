/*
 * \file DQMEventInfo.cc
 * \author M. Zanetti - CERN PH
 * Last Update:
 *
 */
#include "DQMEventInfo.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/ParameterSet/interface/Registry.h"
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
  subsystemname_ = parameters_.getUntrackedParameter<std::string>("subSystemFolder", "YourSubsystem") ;

  eventInfoFolder_ = subsystemname_ + "/" +  folder ;
  evtRateWindow_ = parameters_.getUntrackedParameter<double>("eventRateWindow", 0.5);
  if(evtRateWindow_<=0.15) evtRateWindow_=0.15;

}

DQMEventInfo::~DQMEventInfo(){
}

void DQMEventInfo::bookHistograms(DQMStore::IBooker & ibooker,
                                  edm::Run const & iRun,
                                  edm::EventSetup const & /* iSetup */)
{
  ibooker.setCurrentFolder(eventInfoFolder_) ;

  //Event specific contents
  runId_     = ibooker.bookInt("iRun");
  runId_->Fill(iRun.id().run());
  lumisecId_ = ibooker.bookInt("iLumiSection");
  lumisecId_->Fill(-1);
  eventId_   = ibooker.bookInt("iEvent");
  eventId_->Fill(-1);
  eventTimeStamp_ = ibooker.bookFloat("eventTimeStamp");

  ibooker.setCurrentFolder(eventInfoFolder_) ;
  //Process specific contents
  processTimeStamp_ = ibooker.bookFloat("processTimeStamp");
  processTimeStamp_->Fill(currentTime_);
  processLatency_ = ibooker.bookFloat("processLatency");
  processTimeStamp_->Fill(-1);
  processEvents_ = ibooker.bookInt("processedEvents");
  processEvents_->Fill(pEvent_);
  processEventRate_ = ibooker.bookFloat("processEventRate");
  processEventRate_->Fill(-1);
  nUpdates_= ibooker.bookInt("processUpdates");
  nUpdates_->Fill(-1);

  //Static Contents
  processId_= ibooker.bookInt("processID");
  processId_->Fill(getpid());
  processStartTimeStamp_ = ibooker.bookFloat("processStartTimeStamp");
  processStartTimeStamp_->Fill(currentTime_);
  runStartTimeStamp_ = ibooker.bookFloat("runStartTimeStamp");
  runStartTimeStamp_->Fill(stampToReal(iRun.beginTime()));
  char hostname[65];
  gethostname(hostname,64);
  hostname[64] = 0;
  hostName_= ibooker.bookString("hostName",hostname);
  processName_= ibooker.bookString("processName",subsystemname_);
  char* pwd = getcwd(NULL, 0);
  workingDir_= ibooker.bookString("workingDir",pwd);
  free(pwd);
  cmsswVer_= ibooker.bookString("CMSSW_Version",edm::getReleaseVersion());

  // Folder to be populated by sub-systems' code
  std::string subfolder = eventInfoFolder_ + "/reportSummaryContents" ;
  ibooker.setCurrentFolder(subfolder);

  //Online static histograms
  const edm::ParameterSet &sourcePSet = edm::getProcessParameterSet().getParameterSet("@main_input");
  if (sourcePSet.getParameter<std::string>("@module_type") == "EventStreamHttpReader" ){
    std::string evSelection;
    std::vector<std::string> evSelectionList;
    const edm::ParameterSet &evSelectionPSet = sourcePSet.getUntrackedParameterSet("SelectEvents");
    evSelectionList = evSelectionPSet.getParameter<std::vector<std::string> >("SelectEvents");
    for ( std::vector<std::string>::iterator it = evSelectionList.begin(); it <  evSelectionList.end(); it++ )
      evSelection += "'"+ *it + "', ";

    evSelection.resize(evSelection.length()-2);
    ibooker.setCurrentFolder(eventInfoFolder_);
    ibooker.bookString("eventSelection",evSelection);
  }


}


void DQMEventInfo::beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c)
{
  lumisecId_->Fill(l.id().luminosityBlock());
}

void DQMEventInfo::analyze(const edm::Event& e, const edm::EventSetup& c){

  eventId_->Fill(e.id().event()); // Handing edm::EventNumber_t to Fill method which will handle further casting
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

  return;
}
