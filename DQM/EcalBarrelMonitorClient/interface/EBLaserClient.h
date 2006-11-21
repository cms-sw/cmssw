#ifndef EBLaserClient_H
#define EBLaserClient_H

/*
 * \file EBLaserClient.h
 *
 * $Date: 2006/10/18 16:57:51 $
 * $Revision: 1.39 $
 * \author G. Della Ricca
 *
*/

#include <vector>
#include <string>
 
#include "TROOT.h"
#include "TProfile2D.h"
#include "TH1F.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"

#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

using namespace cms;
using namespace edm;
using namespace std;

class EBLaserClient : public EBClient {

public:

/// Constructor
EBLaserClient(const ParameterSet& ps);

/// Destructor
virtual ~EBLaserClient();

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe(void);
void subscribeNew(void);
void unsubscribe(void);

/// Analyze
void analyze(void);

/// BeginJob
void beginJob(MonitorUserInterface* mui);

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
void htmlOutput(int run, string htmlDir, string htmlName);

/// BeginRunDB
void beginRunDb(void);

/// WriteDB
bool writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov, int ism);

/// EndRunDb
void endRunDb(void);

/// Get Functions
 inline int getEvtPerJob() { return ievt_; }
 inline int getEvtPerRun() { return jevt_; }

private:

int ievt_;
int jevt_;

bool collateSources_;
bool cloneME_;
bool enableQT_;

bool verbose_;

bool enableMonitorDaemon_;

string prefixME_;

vector<int> superModules_;

MonitorUserInterface* mui_;

CollateMonitorElement* me_h01_[36];
CollateMonitorElement* me_h02_[36];
CollateMonitorElement* me_h03_[36];
CollateMonitorElement* me_h04_[36];
CollateMonitorElement* me_h05_[36];
CollateMonitorElement* me_h06_[36];
CollateMonitorElement* me_h07_[36];
CollateMonitorElement* me_h08_[36];

CollateMonitorElement* me_hs01_[36];
CollateMonitorElement* me_hs02_[36];
CollateMonitorElement* me_hs03_[36];
CollateMonitorElement* me_hs04_[36];

CollateMonitorElement* me_h09_[36];
CollateMonitorElement* me_h10_[36];
CollateMonitorElement* me_h11_[36];
CollateMonitorElement* me_h12_[36];

TProfile2D* h01_[36];
TProfile2D* h02_[36];
TProfile2D* h03_[36];
TProfile2D* h04_[36];
TProfile2D* h05_[36];
TProfile2D* h06_[36];
TProfile2D* h07_[36];
TProfile2D* h08_[36];

MEContentsProf2DWithinRangeROOT* qth01_[36];
MEContentsProf2DWithinRangeROOT* qth02_[36];
MEContentsProf2DWithinRangeROOT* qth03_[36];
MEContentsProf2DWithinRangeROOT* qth04_[36];

TProfile2D* h09_[36];
TProfile2D* h10_[36];
TProfile2D* h11_[36];
TProfile2D* h12_[36];

TProfile2D* hs01_[36];
TProfile2D* hs02_[36];
TProfile2D* hs03_[36];
TProfile2D* hs04_[36];

MonitorElement* meg01_[36];
MonitorElement* meg02_[36];
MonitorElement* meg03_[36];
MonitorElement* meg04_[36];

MonitorElement* mea01_[36];
MonitorElement* mea02_[36];
MonitorElement* mea03_[36];
MonitorElement* mea04_[36];

MonitorElement* met01_[36];
MonitorElement* met02_[36];
MonitorElement* met03_[36];
MonitorElement* met04_[36];

MonitorElement* meaopn01_[36];
MonitorElement* meaopn02_[36];
MonitorElement* meaopn03_[36];
MonitorElement* meaopn04_[36];

CollateMonitorElement* me_i01_[36];
CollateMonitorElement* me_i02_[36];
CollateMonitorElement* me_i03_[36];
CollateMonitorElement* me_i04_[36];
CollateMonitorElement* me_i05_[36];
CollateMonitorElement* me_i06_[36];
CollateMonitorElement* me_i07_[36];
CollateMonitorElement* me_i08_[36];

TProfile2D* i01_[36];
TProfile2D* i02_[36];
TProfile2D* i03_[36];
TProfile2D* i04_[36];
TProfile2D* i05_[36];
TProfile2D* i06_[36];
TProfile2D* i07_[36];
TProfile2D* i08_[36];

CollateMonitorElement* me_j01_[36];
CollateMonitorElement* me_j02_[36];
CollateMonitorElement* me_j03_[36];
CollateMonitorElement* me_j04_[36];
CollateMonitorElement* me_j05_[36];
CollateMonitorElement* me_j06_[36];
CollateMonitorElement* me_j07_[36];
CollateMonitorElement* me_j08_[36];

TProfile2D* j01_[36];
TProfile2D* j02_[36];
TProfile2D* j03_[36];
TProfile2D* j04_[36];
TProfile2D* j05_[36];
TProfile2D* j06_[36];
TProfile2D* j07_[36];
TProfile2D* j08_[36];

// Quality check on crystals

float percentVariation_;

};

#endif
