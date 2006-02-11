#ifndef EBPedestalTask_H
#define EBPedestalTask_H

/*
 * \file EBPedestalTask.h
 *
 * $Date: 2006/02/05 22:19:22 $
 * $Revision: 1.16 $
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

class EBPedestalTask: public EDAnalyzer{

public:

/// Constructor
EBPedestalTask(const ParameterSet& ps);

/// Destructor
virtual ~EBPedestalTask();

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

MonitorElement* mePedMapG01_[36];
MonitorElement* mePedMapG06_[36];
MonitorElement* mePedMapG12_[36];

MonitorElement* mePed3SumMapG01_[36];
MonitorElement* mePed3SumMapG06_[36];
MonitorElement* mePed3SumMapG12_[36];

MonitorElement* mePed5SumMapG01_[36];
MonitorElement* mePed5SumMapG06_[36];
MonitorElement* mePed5SumMapG12_[36];

MonitorElement* mePnPedMapG01_[36];
MonitorElement* mePnPedMapG16_[36];

bool init_;

};

#endif
