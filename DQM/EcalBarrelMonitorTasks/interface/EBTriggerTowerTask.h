#ifndef EBTriggerTowerTask_H
#define EBTriggerTowerTask_H

/*
 * \file EBTriggerTowerTask.h
 *
 * $Date: 2006/09/13 07:37:45 $
 * $Revision: 1.2 $
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

class EBTriggerTowerTask: public edm::EDAnalyzer{

public:

/// Constructor
EBTriggerTowerTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBTriggerTowerTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(const edm::EventSetup& c);

/// EndJob
void endJob(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

MonitorElement* meEtMap_[36];

MonitorElement* meVeto_[36];

MonitorElement* meFlags_[36];

MonitorElement* meEtMapT_[36][68];
MonitorElement* meEtMapR_[36][68];

bool init_;

};

#endif
