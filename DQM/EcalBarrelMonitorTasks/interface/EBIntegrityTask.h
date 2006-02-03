#ifndef EBIntegrityTask_H
#define EBIntegrityTask_H

/*
 * \file EBIntegrityTask.h
 *
 * $Date: 2006/01/29 17:21:27 $
 * $Revision: 1.6 $
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

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace cms;
using namespace std;

class EBIntegrityTask: public edm::EDAnalyzer{

public:

/// Constructor
EBIntegrityTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBIntegrityTask();

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

MonitorElement* meIntegrityChId[36];
MonitorElement* meIntegrityGain[36];
MonitorElement* meIntegrityGainSwitch[36];
MonitorElement* meIntegrityGainSwitchStay[36];
MonitorElement* meIntegrityTTId[36];
MonitorElement* meIntegrityTTBlockSize[36];
MonitorElement* meIntegrityDCCSize;

ofstream logFile_;

bool init_;

};

#endif
