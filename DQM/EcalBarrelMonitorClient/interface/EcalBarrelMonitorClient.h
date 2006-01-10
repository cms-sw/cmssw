#ifndef EcalBarrelMonitorClient_H
#define EcalBarrelMonitorClient_H

/*
 * \file EcalBarrelMonitorClient.h
 *
 * $Date: 2006/01/07 16:54:43 $
 * $Revision: 1.30 $
 * \author G. Della Ricca
 * \author F. Cossutti
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
#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBIntegrityClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalOnlineClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBTestPulseClient.h>

#include <DQM/EcalBarrelMonitorClient/interface/EBCosmicClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBElectronClient.h>

#include "CalibCalorimetry/EcalDBInterface/interface/EcalCondDBInterface.h"
#include "CalibCalorimetry/EcalDBInterface/interface/RunTag.h"
#include "CalibCalorimetry/EcalDBInterface/interface/RunDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/RunIOV.h"

#include "CalibCalorimetry/EcalDBInterface/interface/MonRunDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonRunIOV.h"

#include "TROOT.h"
#include "TGaxis.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace std;

class EcalBarrelMonitorClient: public edm::EDAnalyzer{

public:

/// Constructor
EcalBarrelMonitorClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EcalBarrelMonitorClient();

protected:

/// Subscribe/Unsubscribe to Monitoring Elements
virtual void subscribe();
virtual void subscribeNew();
virtual void unsubscribe();

/// Analyze
virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
virtual void beginJob(const edm::EventSetup& c);

/// EndJob
virtual void endJob(void);

/// BeginRun
virtual void beginRun(const edm::EventSetup& c);

/// EndRun
virtual void endRun(void);

/// Setup
virtual void setup(void);

/// Cleanup
virtual void cleanup(void);

/// HtmlOutput
virtual void htmlOutput(void);

/// BeginRunDB
virtual void beginRunDb(void);

/// WriteDB
virtual void writeDb(void);

/// EndRunDB
virtual void endRunDb(void);

private:

int ievt_;
int jevt_;

bool collateSources_;

bool verbose_;

string clientName_;
string hostName_;
int hostPort_;

string outputFile_;

string dbName_;
string dbHostName_;
string dbUserName_;
string dbPassword_;

RunIOV runiov_;
MonRunIOV moniov_;
int subrun_;

string baseHtmlDir_;

MonitorUserInterface* mui_;

string location_;
string runtype_;
string status_;
int run_;
int evt_;

bool begin_run_done_;
bool end_run_done_;

bool forced_begin_run_;
bool forced_end_run_;

int last_update_;

int last_jevt_;

int unknowns_;

EBIntegrityClient* integrity_client_;

EBCosmicClient* cosmic_client_;
EBLaserClient* laser_client_;
EBPedestalClient* pedestal_client_;
EBPedestalOnlineClient* pedestalonline_client_;
EBTestPulseClient* testpulse_client_;
EBElectronClient* electron_client_;

CollateMonitorElement* me_h_;

TH1F* h_;

};

#endif
