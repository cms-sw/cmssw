#ifndef DTTestPulsesTask_H
#define DTTestPulsesTask_H

/*
 * \file DTTestPulsesTask.h
 *
 * $Date: 2006/05/24 17:21:37 $
 * $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>

class DTGeometry;
class DTLayerId;
class DTRangeT0;

class DTTestPulsesTask: public edm::EDAnalyzer{

public:
  
  /// Constructor
  DTTestPulsesTask(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTTestPulsesTask();
  
protected:
  
  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Book the ME
  void bookHistos(const DTLayerId& dtLayer, std::string folder, std::string histoTag);
  
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  
private:
  
  int nevents;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;

  edm::ESHandle<DTGeometry> muonGeom;

  edm::ESHandle<DTRangeT0> t0RangeMap;

  std::string outputFile;

  std::pair <int, int> t0sPeakRange;
  
  // My monitor elements
  std::map<int, MonitorElement*> testPulsesProfiles;
  std::map<int, MonitorElement*> testPulsesOccupancies;
  std::map<int, MonitorElement*> testPulsesTimeBoxes;

  
};

#endif
