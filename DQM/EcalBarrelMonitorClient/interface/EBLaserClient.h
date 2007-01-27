#ifndef EBLaserClient_H
#define EBLaserClient_H

/*
 * \file EBLaserClient.h
 *
 * $Date: 2007/01/26 20:26:04 $
 * $Revision: 1.46 $
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

/// softReset
void softReset(void);

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

/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, int ism);

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

CollateMonitorElement* me_h09_[36];
CollateMonitorElement* me_h10_[36];
CollateMonitorElement* me_h11_[36];
CollateMonitorElement* me_h12_[36];

CollateMonitorElement* me_h13_[36];
CollateMonitorElement* me_h14_[36];
CollateMonitorElement* me_h15_[36];
CollateMonitorElement* me_h16_[36];
CollateMonitorElement* me_h17_[36];
CollateMonitorElement* me_h18_[36];
CollateMonitorElement* me_h19_[36];
CollateMonitorElement* me_h20_[36];

CollateMonitorElement* me_h21_[36];
CollateMonitorElement* me_h22_[36];
CollateMonitorElement* me_h23_[36];
CollateMonitorElement* me_h24_[36];

CollateMonitorElement* me_hs01_[36];
CollateMonitorElement* me_hs02_[36];
CollateMonitorElement* me_hs03_[36];
CollateMonitorElement* me_hs04_[36];

CollateMonitorElement* me_hs05_[36];
CollateMonitorElement* me_hs06_[36];
CollateMonitorElement* me_hs07_[36];
CollateMonitorElement* me_hs08_[36];

TProfile2D* h01_[36];
TProfile2D* h02_[36];
TProfile2D* h03_[36];
TProfile2D* h04_[36];
TProfile2D* h05_[36];
TProfile2D* h06_[36];
TProfile2D* h07_[36];
TProfile2D* h08_[36];

TProfile2D* h09_[36];
TProfile2D* h10_[36];
TProfile2D* h11_[36];
TProfile2D* h12_[36];

TProfile2D* h13_[36];
TProfile2D* h14_[36];
TProfile2D* h15_[36];
TProfile2D* h16_[36];
TProfile2D* h17_[36];
TProfile2D* h18_[36];
TProfile2D* h19_[36];
TProfile2D* h20_[36];
 
TProfile2D* h21_[36];
TProfile2D* h22_[36];
TProfile2D* h23_[36];
TProfile2D* h24_[36];

MEContentsProf2DWithinRangeROOT* qth01_[36];
MEContentsProf2DWithinRangeROOT* qth02_[36];
MEContentsProf2DWithinRangeROOT* qth03_[36];
MEContentsProf2DWithinRangeROOT* qth04_[36];
MEContentsProf2DWithinRangeROOT* qth05_[36];
MEContentsProf2DWithinRangeROOT* qth06_[36];
MEContentsProf2DWithinRangeROOT* qth07_[36];
MEContentsProf2DWithinRangeROOT* qth08_[36];

MEContentsProf2DWithinRangeROOT* qth09_[36];
MEContentsProf2DWithinRangeROOT* qth10_[36];
MEContentsProf2DWithinRangeROOT* qth11_[36];
MEContentsProf2DWithinRangeROOT* qth12_[36];
MEContentsProf2DWithinRangeROOT* qth13_[36];
MEContentsProf2DWithinRangeROOT* qth14_[36];
MEContentsProf2DWithinRangeROOT* qth15_[36];
MEContentsProf2DWithinRangeROOT* qth16_[36];

TProfile2D* hs01_[36];
TProfile2D* hs02_[36];
TProfile2D* hs03_[36];
TProfile2D* hs04_[36];

TProfile2D* hs05_[36];
TProfile2D* hs06_[36];
TProfile2D* hs07_[36];
TProfile2D* hs08_[36];

MonitorElement* meg01_[36];
MonitorElement* meg02_[36];
MonitorElement* meg03_[36];
MonitorElement* meg04_[36];

MonitorElement* meg05_[36];
MonitorElement* meg06_[36];
MonitorElement* meg07_[36];
MonitorElement* meg08_[36];

MonitorElement* mea01_[36];
MonitorElement* mea02_[36];
MonitorElement* mea03_[36];
MonitorElement* mea04_[36];
MonitorElement* mea05_[36];
MonitorElement* mea06_[36];
MonitorElement* mea07_[36];
MonitorElement* mea08_[36];

MonitorElement* met01_[36];
MonitorElement* met02_[36];
MonitorElement* met03_[36];
MonitorElement* met04_[36];
MonitorElement* met05_[36];
MonitorElement* met06_[36];
MonitorElement* met07_[36];
MonitorElement* met08_[36];

MonitorElement* meaopn01_[36];
MonitorElement* meaopn02_[36];
MonitorElement* meaopn03_[36];
MonitorElement* meaopn04_[36];
MonitorElement* meaopn05_[36];
MonitorElement* meaopn06_[36];
MonitorElement* meaopn07_[36];
MonitorElement* meaopn08_[36];

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

// Quality check on PNs

float meanThresholdPN_;
float amplitudeThresholdPN_;

};

#endif
