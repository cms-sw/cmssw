#ifndef EcalBarrelMonitorModule_H
#define EcalBarrelMonitorModule_H

/*
 * \file EcalBarrelMonitorModule.h
 *
 * $Date: 2005/11/10 16:45:05 $
 * $Revision: 1.15 $
 * \author G. Della Ricca
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

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBCosmicTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBLaserTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBPedestalTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBPedPreSampleTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBTestPulseTask.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace std;

class EcalBarrelMonitorModule: public edm::EDAnalyzer{

public:

/// Constructor
EcalBarrelMonitorModule(const edm::ParameterSet& ps);

/// Destructor
virtual ~EcalBarrelMonitorModule();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:

int evtType;
int runType;

int ievt;
int irun;

DaqMonitorBEInterface* dbe;

MonitorElement* meStatus;

MonitorElement* meRun;
MonitorElement* meEvt;

MonitorElement* meEvtType;
MonitorElement* meRunType;

MonitorElement* meEBdigi;
MonitorElement* meEBhits;

MonitorElement* meEvent[36];

EBCosmicTask* cosmic_task;
EBLaserTask* laser_task;
EBPedestalTask* pedestal_task;
EBPedPreSampleTask* pedpresample_task;
EBTestPulseTask* testpulse_task;

string outputFile;

ofstream logFile;

};

#endif
