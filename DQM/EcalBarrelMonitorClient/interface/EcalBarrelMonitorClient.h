#ifndef EcalBarrelMonitorClient_H
#define EcalBarrelMonitorClient_H

/*
 * \file EcalBarrelMonitorClient.h
 *
 * $Date: 2005/11/16 08:36:44 $
 * $Revision: 1.12 $
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

#include <DQM/EcalBarrelMonitorClient/interface/EBIntegrityClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBPedPreSampleClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBTestPulseClient.h>

#include "CalibCalorimetry/EcalDBInterface/interface/EcalCondDBInterface.h"
#include "CalibCalorimetry/EcalDBInterface/interface/RunTag.h"
#include "CalibCalorimetry/EcalDBInterface/interface/RunIOV.h"

#include "TROOT.h"

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

// BeginJob
virtual void beginJob(const edm::EventSetup& c);

// EndJob
virtual void endJob(void);

// BeginRun
virtual void beginRun(const edm::EventSetup& c);

// EndRun
virtual void endRun(void);

// HtmlOutput
virtual void htmlOutput(void);

private:

int ievt_;
int jevt_;

int last_update_;

int last_jevt_;

MonitorUserInterface* mui_;

string clientName_;
string hostName_;
int hostPort_;

EcalCondDBInterface* econn_;

string dbName_;
string dbHostName_;
string dbUserName_;
string dbPassword_;

RunIOV* runiov_;
RunTag* runtag_;

string location_;
string runtype_;
string status_;
int run_;
int evt_;

string baseHtmlDir_;

EBIntegrityClient* integrity_client_;
EBLaserClient* laser_client_;
EBPedestalClient* pedestal_client_;
EBPedPreSampleClient* pedpresample_client_;
EBTestPulseClient* testpulse_client_;

TH1F* h_;

};

#endif
