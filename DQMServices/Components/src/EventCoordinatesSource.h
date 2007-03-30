#ifndef EventCoordinatesSource_H
#define EventCoordinatesSource_H

/*
 * \file EventCoordinatesSource.h
 *
 * $Date: 2007/03/29 14:52:55 $
 * $Revision: 1.2 $
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



class EventCoordinatesSource: public edm::EDAnalyzer{

public:

  /// Constructor
  EventCoordinatesSource(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~EventCoordinatesSource();

protected:

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);


private:

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;

  MonitorElement * runId;
  MonitorElement * eventId;
  MonitorElement * timeStamp;

};

#endif
