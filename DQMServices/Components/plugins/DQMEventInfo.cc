/*
 * \file DQMEventInfo.h
 *
 * \author M. Zanetti - INFN Padova
 *
*/
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <cstdio>
#include <cmath>
#include <map>

#include <sys/time.h>
#include <TSystem.h>

#include <boost/algorithm/string/join.hpp>

class DQMEventInfo : public DQMOneEDAnalyzer<> {
public:
  /// Constructor
  DQMEventInfo(const edm::ParameterSet& ps);

  /// Destructor
  ~DQMEventInfo() override = default;

protected:
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  std::string eventInfoFolder_;
  std::string subsystemname_;

  double currentTime_, lastUpdateTime_, lastAvgTime_;
  double runStartTime_;
  double evtRateWindow_;
  int64_t evtRateCount_;
  int64_t pEvent_;

  //////////////////////////////////////////////////////////////////
  ///These MEs are filled with the info from the most recent event
  ///   by the module
  //////////////////////////////////////////////////////////////////
  MonitorElement* runId_;
  MonitorElement* runStartTimeStamp_;  ///UTC time of the run start
  MonitorElement* eventId_;
  MonitorElement* lumisecId_;
  MonitorElement* eventTimeStamp_;

  //////////////////////////////////////////////////////////////////
  ///These MEs are either static or updated upon each analyze() call
  //////////////////////////////////////////////////////////////////
  MonitorElement* nUpdates_;               ///Number of collector updates (TBD)
  MonitorElement* processId_;              ///The PID associated with this job
  MonitorElement* processStartTimeStamp_;  ///The UTC time of the first event processed
  MonitorElement* processTimeStamp_;       ///The UTC time of the last event
  MonitorElement* processLatency_;         ///Time elapsed since the last event
  MonitorElement* processEventRate_;       ///Avg # of events in programmable window (default: 5 min)
  MonitorElement* processEvents_;          ///# of event processed so far
  MonitorElement* hostName_;               ///Hostname of the local machine
  MonitorElement* processName_;            ///DQM "name" of the job (eg, Hcal or DT)
  MonitorElement* workingDir_;             ///Current working directory of the job
  MonitorElement* cmsswVer_;               ///CMSSW version run for this job
  MonitorElement* dqmPatch_;               ///DQM patch version for this job
  MonitorElement* errSummary_;             ///Subdetector-specific error summary (float)
  MonitorElement* errSummaryEtaPhi_;       ///Subdetector-specific etaPhi summary (float)
  MonitorElement* errSummarySegment_[10];
};

static inline double stampToReal(edm::Timestamp time) {
  return (time.value() >> 32) + 1e-6 * (time.value() & 0xffffffff);
}

static inline double stampToReal(const timeval& time) { return time.tv_sec + 1e-6 * time.tv_usec; }

DQMEventInfo::DQMEventInfo(const edm::ParameterSet& ps) {
  struct timeval now;
  gettimeofday(&now, nullptr);

  pEvent_ = 0;
  evtRateCount_ = 0;
  lastAvgTime_ = currentTime_ = stampToReal(now);

  // read config parms
  std::string folder = ps.getUntrackedParameter<std::string>("eventInfoFolder", "EventInfo");
  subsystemname_ = ps.getUntrackedParameter<std::string>("subSystemFolder", "YourSubsystem");

  eventInfoFolder_ = subsystemname_ + "/" + folder;
  evtRateWindow_ = ps.getUntrackedParameter<double>("eventRateWindow", 0.5);
  if (evtRateWindow_ <= 0.15)
    evtRateWindow_ = 0.15;
}

void DQMEventInfo::bookHistograms(DQMStore::IBooker& ibooker,
                                  edm::Run const& iRun,
                                  edm::EventSetup const& /* iSetup */) {
  ibooker.setCurrentFolder(eventInfoFolder_);

  //Event specific contents
  runId_ = ibooker.bookInt("iRun");
  runId_->Fill(iRun.id().run());
  lumisecId_ = ibooker.bookInt("iLumiSection");
  lumisecId_->Fill(-1);
  eventId_ = ibooker.bookInt("iEvent");
  eventId_->Fill(-1);
  eventTimeStamp_ = ibooker.bookFloat("eventTimeStamp");

  ibooker.setCurrentFolder(eventInfoFolder_);
  //Process specific contents
  processTimeStamp_ = ibooker.bookFloat("processTimeStamp");
  processTimeStamp_->Fill(currentTime_);
  processLatency_ = ibooker.bookFloat("processLatency");
  processTimeStamp_->Fill(-1);
  processEvents_ = ibooker.bookInt("processedEvents");
  processEvents_->Fill(pEvent_);
  processEventRate_ = ibooker.bookFloat("processEventRate");
  processEventRate_->Fill(-1);
  nUpdates_ = ibooker.bookInt("processUpdates");
  nUpdates_->Fill(-1);

  //Static Contents
  processId_ = ibooker.bookInt("processID");
  processId_->Fill(getpid());
  processStartTimeStamp_ = ibooker.bookFloat("processStartTimeStamp");
  processStartTimeStamp_->Fill(currentTime_);
  runStartTimeStamp_ = ibooker.bookFloat("runStartTimeStamp");
  runStartTimeStamp_->Fill(stampToReal(iRun.beginTime()));
  char hostname[65];
  gethostname(hostname, 64);
  hostname[64] = 0;
  hostName_ = ibooker.bookString("hostName", hostname);
  processName_ = ibooker.bookString("processName", subsystemname_);
  char* pwd = getcwd(nullptr, 0);
  workingDir_ = ibooker.bookString("workingDir", pwd);
  free(pwd);
  cmsswVer_ = ibooker.bookString("CMSSW_Version", edm::getReleaseVersion());

  // Folder to be populated by sub-systems' code
  std::string subfolder = eventInfoFolder_ + "/reportSummaryContents";
  ibooker.setCurrentFolder(subfolder);

  //Online static histograms
  const edm::ParameterSet& sourcePSet =
      edm::getProcessParameterSetContainingModule(moduleDescription()).getParameterSet("@main_input");

  if (sourcePSet.getParameter<std::string>("@module_type") == "DQMStreamerReader") {
    std::string evSelection;
    std::vector<std::string> evSelectionList;
    std::string delimiter(", ");
    evSelectionList = sourcePSet.getUntrackedParameter<std::vector<std::string> >("SelectEvents");
    // add single quotes inline in the vector of HLT paths:
    // we do copy assignment, and getUntrackedParameter returns
    // a by-value copy of the vector of strings
    std::for_each(evSelectionList.begin(), evSelectionList.end(), [](std::string& s) {
      std::string squote("'");
      s = squote + s + squote;
    });
    evSelection = boost::algorithm::join(evSelectionList, delimiter);
    // if no HLT paths are specified, no selections are performed:
    // we mark this with an asterisk.
    if (evSelection.empty()) {
      evSelection = std::string("'*'");
    }
    ibooker.setCurrentFolder(eventInfoFolder_);
    ibooker.bookString("eventSelection", evSelection);
  }
}

void DQMEventInfo::analyze(const edm::Event& e, const edm::EventSetup& c) {
  //Filling lumi here guarantees that the lumi number corresponds to the event when
  // using multiple concurrent lumis in a job
  lumisecId_->Fill(e.id().luminosityBlock());
  eventId_->Fill(e.id().event());  // Handing edm::EventNumber_t to Fill method which will handle further casting
  eventTimeStamp_->Fill(stampToReal(e.time()));

  pEvent_++;
  evtRateCount_++;
  processEvents_->Fill(pEvent_);

  struct timeval now;
  gettimeofday(&now, nullptr);
  lastUpdateTime_ = currentTime_;
  currentTime_ = stampToReal(now);

  processTimeStamp_->Fill(currentTime_);
  processLatency_->Fill(currentTime_ - lastUpdateTime_);

  double delta = currentTime_ - lastAvgTime_;
  if (delta >= (evtRateWindow_ * 60.0)) {
    processEventRate_->Fill(evtRateCount_ / delta);
    evtRateCount_ = 0;
    lastAvgTime_ = currentTime_;
  }

  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMEventInfo);
