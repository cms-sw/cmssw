#ifndef DQMEventInfo_H
#define DQMEventInfo_H

/*
 * \file DQMEventInfo.h
 *
 * $Date: 2007/11/21 00:19:12 $
 * $Revision: 1.4 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/ServiceRegistry/interface/Service.h>

#include <DQMServices/Core/interface/DaqMonitorBEInterface.h>
#include <DQMServices/Core/interface/MonitorElementBaseT.h>
#include <DQMServices/Daemon/interface/MonitorDaemon.h>

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


private:

  double getUTCtime(timeval* a, timeval* b = NULL);

  DaqMonitorBEInterface* dbe_;

  edm::ParameterSet parameters_;
  timeval currentTime_, lastUpdateTime_, lastAvgTime_;
  float evtRateWindow_;
  int evtRateCount_;
  int pEvent_;

  //////////////////////////////////////////////////////////////////
  ///These MEs are filled with the info from the most recent event 
  ///   by the module
  //////////////////////////////////////////////////////////////////
  MonitorElement * runId_;
  MonitorElement * eventId_;
  MonitorElement * lumisecId_;
  MonitorElement * eventTimeStamp_;
  
  //////////////////////////////////////////////////////////////////
  ///These MEs are either static or updated upon each analyze() call
  //////////////////////////////////////////////////////////////////
  MonitorElement * nUpdates_;          ///Number of collector updates (TBD)
  MonitorElement * processId_;         ///The PID associated with this job
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

};

#endif
