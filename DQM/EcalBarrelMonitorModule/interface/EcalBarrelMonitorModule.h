#ifndef EcalBarrelMonitorModule_H
#define EcalBarrelMonitorModule_H

/*
 * \file EcalBarrelMonitorModule.h
 *
 * $Date: 2005/12/23 08:57:15 $
 * $Revision: 1.23 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQM/EcalBarrelMonitorTasks/interface/EBIntegrityTask.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBCosmicTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBLaserTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBPnDiodeTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBPedestalTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBPedestalOnlineTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBTestPulseTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBElectronTask.h>

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

int evtType_;
int runType_;

int ievt_;
int irun_;

bool verbose_;

DaqMonitorBEInterface* dbe_;

MonitorElement* meStatus_;

MonitorElement* meRun_;
MonitorElement* meEvt_;

MonitorElement* meEvtType_;
MonitorElement* meRunType_;

MonitorElement* meEBdigi_;
MonitorElement* meEBhits_;

MonitorElement* meEvent_[36];

EBIntegrityTask* integrity_task_;

EBCosmicTask* cosmic_task_;
EBLaserTask* laser_task_;
EBPnDiodeTask* pndiode_task_;
EBPedestalTask* pedestal_task_;
EBPedestalOnlineTask* pedestalonline_task_;
EBTestPulseTask* testpulse_task_;
EBElectronTask* electron_task_;

string outputFile_;

ofstream logFile_;

};

#endif
