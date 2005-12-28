#ifndef EBTestPulseClient_H
#define EBTestPulseClient_H

/*
 * \file EBTestPulseClient.h
 *
 * $Date: 2005/12/26 13:14:24 $
 * $Revision: 1.13 $
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
#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "CalibCalorimetry/EcalDBInterface/interface/EcalCondDBInterface.h"
#include "CalibCalorimetry/EcalDBInterface/interface/RunTag.h"
#include "CalibCalorimetry/EcalDBInterface/interface/RunIOV.h"

#include "CalibCalorimetry/EcalDBInterface/interface/MonTestPulseDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonPulseShapeDat.h"

#include "TROOT.h"
#include "TStyle.h"
#include "TPaveStats.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace std;

class EBTestPulseClient: public edm::EDAnalyzer{

friend class EcalBarrelMonitorClient;

public:

/// Constructor
EBTestPulseClient(const edm::ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
virtual ~EBTestPulseClient();

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
void writeDb(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag);

private:

int ievt_;
int jevt_;

bool collateSources_;

bool verbose_;

MonitorUserInterface* mui_;

CollateMonitorElement* me_ha01_[36];
CollateMonitorElement* me_ha02_[36];
CollateMonitorElement* me_ha03_[36];

CollateMonitorElement* me_hs01_[36];
CollateMonitorElement* me_hs02_[36];
CollateMonitorElement* me_hs03_[36];

CollateMonitorElement* me_he01_[36];
CollateMonitorElement* me_he02_[36];
CollateMonitorElement* me_he03_[36];

TProfile2D* ha01_[36];
TProfile2D* ha02_[36];
TProfile2D* ha03_[36];

TProfile2D* hs01_[36];
TProfile2D* hs02_[36];
TProfile2D* hs03_[36];

TH2F* he01_[36];
TH2F* he02_[36];
TH2F* he03_[36];

TH2F* g01_[36];
TH2F* g02_[36];
TH2F* g03_[36];

TH1F* a01_[36];
TH1F* a02_[36];
TH1F* a03_[36];

// Quality check on crystals, one per each gain

float amplitudeThreshold_;
float RMSThreshold_;
float threshold_on_AmplitudeErrorsNumber_;

};

#endif
