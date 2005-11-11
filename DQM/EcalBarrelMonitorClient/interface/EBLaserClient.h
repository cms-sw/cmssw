#ifndef EBLaserClient_H
#define EBLaserClient_H

/*
 * \file EBLaserClient.h
 *
 * $Date: 2005/11/10 15:57:22 $
 * $Revision: 1.3 $
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

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "CalibCalorimetry/EcalDBInterface/interface/EcalCondDBInterface.h"
#include "CalibCalorimetry/EcalDBInterface/interface/RunTag.h"
#include "CalibCalorimetry/EcalDBInterface/interface/RunIOV.h"

#include "CalibCalorimetry/EcalDBInterface/interface/MonLaserBlueDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonLaserGreenDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonLaserInfraredDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonLaserRedDat.h"

#include "TROOT.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace std;

class EBLaserClient: public edm::EDAnalyzer{

friend class EcalBarrelMonitorClient;

public:

/// Constructor
EBLaserClient(const edm::ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
virtual ~EBLaserClient();

protected:

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe();
void subscribeNew();
void unsubscribe();

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

// BeginRun
void beginRun(const edm::EventSetup& c);

// EndRun
void endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag);

// HtmlOutput
virtual void htmlOutput(void);

private:

int ievt_;
int jevt_;

MonitorUserInterface* mui_;

};

#endif
