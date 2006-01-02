#ifndef EBIntegrityClient_H
#define EBIntegrityClient_H

/*
 * \file EBIntegrityClient.h
 *
 * $Date: 2006/01/02 09:18:01 $
 * $Revision: 1.19 $
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
#include "CalibCalorimetry/EcalDBInterface/interface/MonRunIOV.h"

#include "CalibCalorimetry/EcalDBInterface/interface/MonCrystalConsistencyDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonTTConsistencyDat.h"

#include "TROOT.h"
#include "TStyle.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace std;

class EBIntegrityClient: public edm::EDAnalyzer{

friend class EcalBarrelMonitorClient;

public:

/// Constructor
EBIntegrityClient(const edm::ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
virtual ~EBIntegrityClient();

protected:

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe();
void subscribeNew();
void unsubscribe();

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(const edm::EventSetup& c);

/// EndJob
void endJob(void);

/// BeginRun
void beginRun(const edm::EventSetup& c);

/// EndRun
void endRun(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

/// HtmlOutput
void htmlOutput(int run, string htmlDir, string htmlName);

/// WriteDB
void writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov);

private:

int ievt_;
int jevt_;

bool collateSources_;

bool verbose_;

MonitorUserInterface* mui_;

CollateMonitorElement* me_h00_;

CollateMonitorElement* me_h01_[36];
CollateMonitorElement* me_h02_[36];
CollateMonitorElement* me_h03_[36];
CollateMonitorElement* me_h04_[36];
CollateMonitorElement* me_h05_[36];
CollateMonitorElement* me_h06_[36];

TH1F* h00_;

TH2F* h01_[36];
TH2F* h02_[36];
TH2F* h03_[36];
TH2F* h04_[36];
TH2F* h05_[36];
TH2F* h06_[36];

TH2F* g01_[36];

CollateMonitorElement* me_h_[36];

TH2F* h_[36];

// Quality criteria for data integrity

float threshCry_;

};

#endif
