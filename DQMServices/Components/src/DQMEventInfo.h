#ifndef DQMEventInfo_H
#define DQMEventInfo_H

/*
 * \file DQMEventInfo.h
 *
 * $Date: 2007/11/05 16:42:13 $
 * $Revision: 1.4 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

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

  MonitorElement * runId_;
  MonitorElement * eventId_;
  MonitorElement * lumisecId_;
  MonitorElement * timeStamp_;

};

#endif
