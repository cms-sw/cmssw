#ifndef DQMEventInfo_H
#define DQMEventInfo_H

/*
 * \file DQMEventInfo.h
 *
 * $Date: 2007/11/15 23:09:41 $
 * $Revision: 1.2 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/CPUTimer.h>
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

  DaqMonitorBEInterface* dbe_;

  edm::ParameterSet parameters_;
  edm::CPUTimer timer_;
  float lastUpdateTime_;
  int pEvent_;

  MonitorElement * runId_;
  MonitorElement * eventId_;
  MonitorElement * lumisecId_;
  MonitorElement * eventTimeStamp_;
  
  MonitorElement * nUpdates_;
  MonitorElement * processId_;
  MonitorElement * processTimeStamp_;
  MonitorElement * processEvents_;
  MonitorElement * hostName_;
  MonitorElement * processName_;
  MonitorElement * workingDir_;
  MonitorElement * cmsswVer_;
  MonitorElement * dqmPatch_;
  MonitorElement * errSummary_;

};

#endif
