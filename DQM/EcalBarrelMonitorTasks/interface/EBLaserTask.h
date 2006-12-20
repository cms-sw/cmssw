#ifndef EBLaserTask_H
#define EBLaserTask_H

/*
 * \file EBLaserTask.h
 *
 * $Date: 2006/06/17 10:07:47 $
 * $Revision: 1.20 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace cms;
using namespace edm;
using namespace std;

class EBLaserTask: public EDAnalyzer{

public:

/// Constructor
EBLaserTask(const ParameterSet& ps);

/// Destructor
virtual ~EBLaserTask();

protected:

/// Analyze
void analyze(const Event& e, const EventSetup& c);

/// BeginJob
void beginJob(const EventSetup& c);

/// EndJob
void endJob(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

MonitorElement* meShapeMapL1A_[36];
MonitorElement* meAmplMapL1A_[36];
MonitorElement* meTimeMapL1A_[36];
MonitorElement* meAmplPNMapL1A_[36];
MonitorElement* meShapeMapL1B_[36];
MonitorElement* meAmplMapL1B_[36];
MonitorElement* meTimeMapL1B_[36];
MonitorElement* meAmplPNMapL1B_[36];
MonitorElement* mePnAmplMapG01L1_[36];
MonitorElement* mePnPedMapG01L1_[36];
MonitorElement* mePnAmplMapG16L1_[36];
MonitorElement* mePnPedMapG16L1_[36];

MonitorElement* meShapeMapL2A_[36];
MonitorElement* meAmplMapL2A_[36];
MonitorElement* meTimeMapL2A_[36];
MonitorElement* meAmplPNMapL2A_[36];
MonitorElement* meShapeMapL2B_[36];
MonitorElement* meAmplMapL2B_[36];
MonitorElement* meTimeMapL2B_[36];
MonitorElement* meAmplPNMapL2B_[36];
MonitorElement* mePnAmplMapG01L2_[36];
MonitorElement* mePnPedMapG01L2_[36];
MonitorElement* mePnAmplMapG16L2_[36];
MonitorElement* mePnPedMapG16L2_[36];

MonitorElement* meShapeMapL3A_[36];
MonitorElement* meAmplMapL3A_[36];
MonitorElement* meTimeMapL3A_[36];
MonitorElement* meAmplPNMapL3A_[36];
MonitorElement* meShapeMapL3B_[36];
MonitorElement* meAmplMapL3B_[36];
MonitorElement* meTimeMapL3B_[36];
MonitorElement* meAmplPNMapL3B_[36];
MonitorElement* mePnAmplMapG01L3_[36];
MonitorElement* mePnPedMapG01L3_[36];
MonitorElement* mePnAmplMapG16L3_[36];
MonitorElement* mePnPedMapG16L3_[36];

MonitorElement* meShapeMapL4A_[36];
MonitorElement* meAmplMapL4A_[36];
MonitorElement* meTimeMapL4A_[36];
MonitorElement* meAmplPNMapL4A_[36];
MonitorElement* meShapeMapL4B_[36];
MonitorElement* meAmplMapL4B_[36];
MonitorElement* meTimeMapL4B_[36];
MonitorElement* meAmplPNMapL4B_[36];
MonitorElement* mePnAmplMapG01L4_[36];
MonitorElement* mePnPedMapG01L4_[36];
MonitorElement* mePnAmplMapG16L4_[36];
MonitorElement* mePnPedMapG16L4_[36];

bool init_;

};

#endif
