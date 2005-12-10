#ifndef EBIntegrityTask_H
#define EBIntegrityTask_H

/*
 * \file EBIntegrityTask.h
 *
 * $Date: 2005/12/07 07:28:53 $
 * $Revision: 1.1 $
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

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace cms;
using namespace std;

class EBIntegrityTask: public edm::EDAnalyzer{

friend class EcalBarrelMonitorModule;

public:

/// Constructor
EBIntegrityTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);

/// Destructor
virtual ~EBIntegrityTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:

int ievt_;

MonitorElement* meIntegrityChId[36];  
MonitorElement* meIntegrityGain[36];
MonitorElement* meIntegrityTTId[36];
MonitorElement* meIntegrityTTBlockSize[36];
MonitorElement* meIntegrityDCCSize;

ofstream logFile_;

};

#endif
