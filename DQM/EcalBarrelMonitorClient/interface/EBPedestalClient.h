#ifndef EBPedestalClient_H
#define EBPedestalClient_H

/*
 * \file EBPedestalClient.h
 *
 * $Date: 2006/03/05 09:50:40 $
 * $Revision: 1.27 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

#include "OnlineDB/EcalCondDB/interface/MonPedestalsDat.h"

#include "OnlineDB/EcalCondDB/interface/MonPNPedDat.h"

#include "TROOT.h"
#include "TStyle.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace edm;
using namespace std;

class EBPedestalClient{

public:

/// Constructor
EBPedestalClient(const ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
virtual ~EBPedestalClient();

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe(void);
void subscribeNew(void);
void unsubscribe(void);

/// Analyze
void analyze(void);

// BeginJob
void beginJob(void);

// EndJob
void endJob(void);

// BeginRun
void beginRun(void);

// EndRun
void endRun(void);

/// Setup
void setup(void);

// Cleanup
void cleanup(void);

// HtmlOutput
void htmlOutput(int run, int jsm, string htmlDir, string htmlName);

// WriteDB
void writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov);

private:

int ievt_;
int jevt_;

bool collateSources_;
bool cloneME_;

bool verbose_;

bool enableMonitorDaemon_;

MonitorUserInterface* mui_;

CollateMonitorElement* me_h01_[36];
CollateMonitorElement* me_h02_[36];
CollateMonitorElement* me_h03_[36];

CollateMonitorElement* me_j01_[36];
CollateMonitorElement* me_j02_[36];
CollateMonitorElement* me_j03_[36];

CollateMonitorElement* me_k01_[36];
CollateMonitorElement* me_k02_[36];
CollateMonitorElement* me_k03_[36];

TProfile2D* h01_[36];
TProfile2D* h02_[36];
TProfile2D* h03_[36];

TProfile2D* j01_[36];
TProfile2D* j02_[36];
TProfile2D* j03_[36];

TProfile2D* k01_[36];
TProfile2D* k02_[36];
TProfile2D* k03_[36];

TH2F* g01_[36];
TH2F* g02_[36];
TH2F* g03_[36];

TH1F* p01_[36];
TH1F* p02_[36];
TH1F* p03_[36];

TH1F* r01_[36];
TH1F* r02_[36];
TH1F* r03_[36];

TH2F* s01_[36];
TH2F* s02_[36];
TH2F* s03_[36];

TH2F* t01_[36];
TH2F* t02_[36];
TH2F* t03_[36];

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
