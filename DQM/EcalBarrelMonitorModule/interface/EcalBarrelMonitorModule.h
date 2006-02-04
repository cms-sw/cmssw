#ifndef EcalBarrelMonitorModule_H
#define EcalBarrelMonitorModule_H

/*
 * \file EcalBarrelMonitorModule.h
 *
 * $Date: 2006/02/02 12:54:54 $
 * $Revision: 1.28 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRecHit/interface/EcalRecHitCollections.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

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

bool enableMonitorDaemon_;

DaqMonitorBEInterface* dbe_;

MonitorElement* meStatus_;

MonitorElement* meRun_;
MonitorElement* meEvt_;

MonitorElement* meEvtType_;
MonitorElement* meRunType_;

MonitorElement* meEBdigi_;
MonitorElement* meEBhits_;

MonitorElement* meEvent_[36];
MonitorElement* meOccupancy_[36];

string outputFile_;

ofstream logFile_;

};

#endif
