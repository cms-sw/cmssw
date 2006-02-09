#ifndef EBLaserTask_H
#define EBLaserTask_H

/*
 * \file EBLaserTask.h
 *
 * $Date: 2006/02/05 22:19:22 $
 * $Revision: 1.17 $
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

private:

int ievt_;

MonitorElement* meShapeMapL1_[36];
MonitorElement* meAmplMapL1_[36];
MonitorElement* meTimeMapL1_[36];
MonitorElement* meAmplPNMapL1_[36];
MonitorElement* mePnAmplMapG01L1_[36];
MonitorElement* mePnPedMapG01L1_[36];
MonitorElement* mePnAmplMapG16L1_[36];
MonitorElement* mePnPedMapG16L1_[36];

MonitorElement* meShapeMapL2_[36];
MonitorElement* meAmplMapL2_[36];
MonitorElement* meTimeMapL2_[36];
MonitorElement* meAmplPNMapL2_[36];
MonitorElement* mePnAmplMapG01L2_[36];
MonitorElement* mePnPedMapG01L2_[36];
MonitorElement* mePnAmplMapG16L2_[36];
MonitorElement* mePnPedMapG16L2_[36];

MonitorElement* meShapeMapL3_[36];
MonitorElement* meAmplMapL3_[36];
MonitorElement* meTimeMapL3_[36];
MonitorElement* meAmplPNMapL3_[36];
MonitorElement* mePnAmplMapG01L3_[36];
MonitorElement* mePnPedMapG01L3_[36];
MonitorElement* mePnAmplMapG16L3_[36];
MonitorElement* mePnPedMapG16L3_[36];

MonitorElement* meShapeMapL4_[36];
MonitorElement* meAmplMapL4_[36];
MonitorElement* meTimeMapL4_[36];
MonitorElement* meAmplPNMapL4_[36];
MonitorElement* mePnAmplMapG01L4_[36];
MonitorElement* mePnPedMapG01L4_[36];
MonitorElement* mePnAmplMapG16L4_[36];
MonitorElement* mePnPedMapG16L4_[36];

bool init_;

};

#endif
