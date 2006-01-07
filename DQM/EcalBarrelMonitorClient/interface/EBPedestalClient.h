#ifndef EBPedestalClient_H
#define EBPedestalClient_H

/*
 * \file EBPedestalClient.h
 *
 * $Date: 2006/01/02 09:18:01 $
 * $Revision: 1.19 $
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
#include "CalibCalorimetry/EcalDBInterface/interface/MonRunIOV.h"

#include "CalibCalorimetry/EcalDBInterface/interface/MonPedestalsDat.h"

#include "CalibCalorimetry/EcalDBInterface/interface/MonPNPedDat.h"

#include "TROOT.h"
#include "TStyle.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace std;

class EBPedestalClient: public edm::EDAnalyzer{

friend class EcalBarrelMonitorClient;

public:

/// Constructor
EBPedestalClient(const edm::ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
virtual ~EBPedestalClient();

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
void endRun(void);

/// Setup
void setup(void);

// Cleanup
void cleanup(void);

// HtmlOutput
void htmlOutput(int run, string htmlDir, string htmlName);

// WriteDB
void writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov);

private:

int ievt_;
int jevt_;

bool collateSources_;

bool verbose_;

MonitorUserInterface* mui_;

CollateMonitorElement* me_h01_[36];
CollateMonitorElement* me_h02_[36];
CollateMonitorElement* me_h03_[36];

TProfile2D* h01_[36];
TProfile2D* h02_[36];
TProfile2D* h03_[36];

TH2F* g01_[36];
TH2F* g02_[36];
TH2F* g03_[36];

TH1F* p01_[36];
TH1F* p02_[36];
TH1F* p03_[36];

TH1F* r01_[36];
TH1F* r02_[36];
TH1F* r03_[36];

CollateMonitorElement* me_i01_[36];
CollateMonitorElement* me_i02_[36];

TProfile2D* i01_[36];
TProfile2D* i02_[36];

// Quality check on crystals, one per each gain

float expectedMean_[3];
float discrepancyMean_[3];
float RMSThreshold_[3];

};

#endif
