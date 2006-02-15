#ifndef DTGlobalRecoTask_H
#define DTGlobalRecoTask_H

/*
 * \file DTGlobalRecoTask.h
 *
 * $Date: 2005/11/15 17:03:58 $
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

#include <iostream>
#include <fstream>
#include <vector>

using namespace cms;
using namespace std;


class DTGlobalRecoTask: public edm::EDAnalyzer{

friend class DTMonitorModule;

public:

/// Constructor
DTGlobalRecoTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe,
		 const edm::EventSetup& context);

/// Destructor
virtual ~DTGlobalRecoTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:

  int nevents;
  
  // My monitor elements
  
  ofstream logFile;
  
};

#endif
