#ifndef DQMEventInfo_H
#define DQMEventInfo_H

/*
 * \file DQMEventInfo.h
 *
 * $Date: 2010/09/13 12:43:25 $
 * $Revision: 1.17 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Run.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/ServiceRegistry/interface/Service.h>

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sys/time.h>

class DQMEventInfo: public edm::EDAnalyzer{

public:

  /// Constructor
  DQMEventInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DQMEventInfo();

protected:

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void beginRun(const edm::Run& r, const edm::EventSetup& c) ;
  void beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c);

private:

//  double getUTCtime(timeval* a, timeval* b = NULL);

  DQMStore *dbe_;

  edm::ParameterSet parameters_;
  std::string eventInfoFolder_;
  
//  timeval currentTime_, lastUpdateTime_, lastAvgTime_;
//  timeval runStartTime_;
//  float evtRateWindow_;
  double currentTime_, lastUpdateTime_, lastAvgTime_;
  double runStartTime_;
  double evtRateWindow_;
  int64_t evtRateCount_;
  int64_t pEvent_;

  //////////////////////////////////////////////////////////////////
  ///These MEs are filled with the info from the most recent event 
  ///   by the module
  //////////////////////////////////////////////////////////////////
  MonitorElement * runId_;
  MonitorElement * runStartTimeStamp_;  ///UTC time of the run start
  MonitorElement * eventId_;
  MonitorElement * lumisecId_;
  MonitorElement * eventTimeStamp_;

  //////////////////////////////////////////////////////////////////
  ///These MEs are either static or updated upon each analyze() call
  //////////////////////////////////////////////////////////////////
  MonitorElement * nUpdates_;          ///Number of collector updates (TBD)
  MonitorElement * processId_;         ///The PID associated with this job
  MonitorElement * processStartTimeStamp_; ///The UTC time of the first event processed
  MonitorElement * processTimeStamp_;  ///The UTC time of the last event
  MonitorElement * processLatency_;    ///Time elapsed since the last event
  MonitorElement * processEventRate_;  ///Avg # of events in programmable window (default: 5 min)
  MonitorElement * processEvents_;     ///# of event processed so far
  MonitorElement * hostName_;          ///Hostname of the local machine
  MonitorElement * processName_;       ///DQM "name" of the job (eg, Hcal or DT)
  MonitorElement * workingDir_;        ///Current working directory of the job
  MonitorElement * cmsswVer_;          ///CMSSW version run for this job
  MonitorElement * dqmPatch_;          ///DQM patch version for this job
  MonitorElement * errSummary_;        ///Subdetector-specific error summary (float)
  MonitorElement * errSummaryEtaPhi_;  ///Subdetector-specific etaPhi summary (float)
  MonitorElement * errSummarySegment_[10];
};

#endif
