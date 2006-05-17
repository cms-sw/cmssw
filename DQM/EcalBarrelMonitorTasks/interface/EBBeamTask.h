#ifndef EBBeamTask_H
#define EBBeamTask_H

/*
 * \file EBBeamTask.h
 *
 * $Date: 2006/05/11 19:46:09 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 * \author G. Franzoni
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
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "TBDataFormats/EcalTBObjects/interface/EcalTBCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace cms;
using namespace edm;
using namespace std;

class EBBeamTask: public EDAnalyzer{

public:

/// Constructor
EBBeamTask(const ParameterSet& ps);

/// Destructor
virtual ~EBBeamTask();

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

MonitorElement* meHodoOcc_[4];
MonitorElement* meHodoRaw_[4];
MonitorElement* meHodoPosRec_;
MonitorElement* meHodoSloXRec_;
MonitorElement* meHodoSloYRec_;
MonitorElement* meHodoQuaXRec_;
MonitorElement* meHodoQuaYRec_;
MonitorElement* meTDCRec_;
MonitorElement* meEvsXRec_;
MonitorElement* meEvsYRec_;
MonitorElement* meCaloVsHodoXPos_;
MonitorElement* meCaloVsHodoYPos_;
MonitorElement* meCaloVsTDCTime_;

bool init_;

int smId;

};

#endif
