#ifndef DTTestPulsesTask_H
#define DTTestPulsesTask_H

/*
 * \file DTTestPulsesTask.h
 *
 * $Date: 2006/02/02 18:27:31 $
 * $Revision: 1.2 $
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

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>

using namespace cms;
using namespace std;


class DTTestPulsesTask: public edm::EDAnalyzer{

friend class DTMonitorModule;

public:

/// Constructor
  DTTestPulsesTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe,
		   const edm::EventSetup& context);

/// Destructor
virtual ~DTTestPulsesTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:

  int nevents;
  
  pair <int, int> t0sPeakRange;

  // My monitor elements
  map<int, MonitorElement*> testPulsesHistos;

  ofstream logFile;
  
};

#endif
