#ifndef EBLaserTask_H
#define EBLaserTask_H

/*
 * \file EBLaserTask.h
 *
 * $Date: 2005/11/14 08:52:30 $
 * $Revision: 1.11 $
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
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>

#include <iostream>
#include <fstream>
#include <vector>

using namespace cms;
using namespace std;

class EBLaserTask: public edm::EDAnalyzer{

friend class EcalBarrelMonitorModule;

public:

/// Constructor
EBLaserTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);

/// Destructor
virtual ~EBLaserTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:

int ievt_;

MonitorElement* meShapeMapL1_[36];
MonitorElement* meAmplMapL1_[36];
MonitorElement* meAmplPNMapL1_[36];

MonitorElement* meShapeMapL2_[36];
MonitorElement* meAmplMapL2_[36];
MonitorElement* meAmplPNMapL2_[36];

MonitorElement* meShapeMapL3_[36];
MonitorElement* meAmplMapL3_[36];
MonitorElement* meAmplPNMapL3_[36];

MonitorElement* meShapeMapL4_[36];
MonitorElement* meAmplMapL4_[36];
MonitorElement* meAmplPNMapL4_[36];

ofstream logFile_;

};

#endif
