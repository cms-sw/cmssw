#ifndef DTTestPulsesTask_H
#define DTTestPulsesTask_H

/*
 * \file DTTestPulsesTask.h
 *
 * $Date: 2006/02/15 08:24:55 $
 * $Revision: 1.1 $
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
class DTTtrig;

using namespace cms;
using namespace std;


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
  void bookHistos(const DTLayerId& dtLayer, string folder, string histoTag);
  
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  
private:
  
  int nevents;

  // TestPules tTrig from the DB
  int tTrig_TP;
  
  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;

  edm::ESHandle<DTGeometry> muonGeom;

  edm::ESHandle<DTTtrig> tTrig_TPMap;

  string outputFile;

  pair <int, int> t0sPeakRange;
  
  // My monitor elements
  map<int, MonitorElement*> testPulsesHistos;
  map<int, MonitorElement*> testPulsesTimeBoxes;

  
};

#endif
