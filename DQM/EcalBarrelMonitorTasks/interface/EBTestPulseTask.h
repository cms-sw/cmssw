#ifndef EBTestPulseTask_H
#define EBTestPulseTask_H

/*
 * \file EBTestPulseTask.h
 *
 * $Date: 2006/01/29 17:21:27 $
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
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>

#include <iostream>
#include <fstream>
#include <vector>

using namespace cms;
using namespace std;

class EBTestPulseTask: public edm::EDAnalyzer{

public:

/// Constructor
EBTestPulseTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBTestPulseTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(const edm::EventSetup& c);

/// EndJob
void endJob(void);

/// Setup
void setup(void);

private:

int ievt_;

MonitorElement* meShapeMapG01_[36];
MonitorElement* meShapeMapG06_[36];
MonitorElement* meShapeMapG12_[36];

MonitorElement* meAmplMapG01_[36];
MonitorElement* meAmplMapG06_[36];
MonitorElement* meAmplMapG12_[36];

MonitorElement* meAmplErrorMapG01_[36];
MonitorElement* meAmplErrorMapG06_[36];
MonitorElement* meAmplErrorMapG12_[36];

MonitorElement* mePnAmplMapG01_[36];
MonitorElement* mePnAmplMapG16_[36];

MonitorElement* mePnPedMapG01_[36];
MonitorElement* mePnPedMapG16_[36];

ofstream logFile_;

// Quality check on crystals, one per each gain

float amplitudeThreshold_;

bool init_;

};

#endif
