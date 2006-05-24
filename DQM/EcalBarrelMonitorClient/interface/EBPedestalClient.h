#ifndef EBPedestalClient_H
#define EBPedestalClient_H

/*
 * \file EBPedestalClient.h
 *
 * $Date: 2006/05/23 09:06:50 $
 * $Revision: 1.30 $
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
void htmlOutput(int run, const std::vector<int> & superModules, string htmlDir, string htmlName);

// WriteDB
void writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov, int ism);

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

MonitorElement* meg01_[36];
MonitorElement* meg02_[36];
MonitorElement* meg03_[36];

MonitorElement* mep01_[36];
MonitorElement* mep02_[36];
MonitorElement* mep03_[36];

MonitorElement* mer01_[36];
MonitorElement* mer02_[36];
MonitorElement* mer03_[36];

MonitorElement* mes01_[36];
MonitorElement* mes02_[36];
MonitorElement* mes03_[36];

MonitorElement* met01_[36];
MonitorElement* met02_[36];
MonitorElement* met03_[36];

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
