#ifndef DTMonitorModule_H
#define DTMonitorModule_H

/*
 * \file DTMonitorModule.h
 *
 * $Date: 2005/11/15 14:02:45 $
 * $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 *
*/
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
 
class DTDigiTask;
class DTTestPulsesTask;
class DTLocalRecoTask;
class DTGlobalRecoTask;
class DTLocalTriggerTask;

using namespace cms;
using namespace std;


class DTMonitorModule: public edm::EDAnalyzer{

public:

  /// Constructor
  DTMonitorModule(const edm::ParameterSet& ps);

  
  /// Destructor
  virtual ~DTMonitorModule();
  
protected:
  
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  // BeginJob
  void beginJob(const edm::ParameterSet& ps, const edm::EventSetup& c);
  
  // EndJob
  void endJob(void);
  
private:
  
  int nevents;

  string outputFile;
  ofstream logFile;
  
  DaqMonitorBEInterface* dbe;
  
  MonitorElement* meStatus;
  MonitorElement* meRun;
  MonitorElement* meEvt;
  MonitorElement* meRunType;
  

};

DEFINE_FWK_MODULE(DTMonitorModule)
  
#endif
