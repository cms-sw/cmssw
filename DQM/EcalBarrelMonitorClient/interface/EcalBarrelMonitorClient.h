#ifndef EcalBarrelMonitorClient_H
#define EcalBarrelMonitorClient_H

/*
 * \file EcalBarrelMonitorClient.h
 *
 * $Date: 2006/02/05 22:21:53 $
 * $Revision: 1.35 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBIntegrityClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalOnlineClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBTestPulseClient.h>

#include <DQM/EcalBarrelMonitorClient/interface/EBCosmicClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBBeamClient.h>

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

class EcalBarrelMonitorClient: public EDAnalyzer{

public:

/// Constructor
EcalBarrelMonitorClient(const ParameterSet& ps);
EcalBarrelMonitorClient(const ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
~EcalBarrelMonitorClient();

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe(void);
void subscribeNew(void);
void unsubscribe(void);

// Initialize
void initialize(const ParameterSet& ps);

/// Analyze
void analyze(void);
void analyze(const Event& e, const EventSetup& c){ this->analyze(); }

/// BeginJob
void beginJob(void);
void beginJob(const EventSetup& c){ this->beginJob(); }

/// EndJob
void endJob(void);

/// BeginRun
void beginRun(void);

/// EndRun
void endRun(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

/// HtmlOutput
void htmlOutput(void);

/// BeginRunDB
void beginRunDb(void);

/// WriteDB
void writeDb(void);

/// EndRunDB
void endRunDb(void);

private:

int ievt_;
int jevt_;

bool collateSources_;
bool cloneME_;

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

bool enableSubRun_;
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

bool enableExit_;

int last_update_;

int last_jevt_;

int unknowns_;

EBIntegrityClient* integrity_client_;

EBCosmicClient* cosmic_client_;
EBLaserClient* laser_client_;
EBPedestalClient* pedestal_client_;
EBPedestalOnlineClient* pedestalonline_client_;
EBTestPulseClient* testpulse_client_;
EBBeamClient* beam_client_;

CollateMonitorElement* me_h_;

TH1F* h_;

};

#endif
