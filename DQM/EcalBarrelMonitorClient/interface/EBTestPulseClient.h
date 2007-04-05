#ifndef EBTestPulseClient_H
#define EBTestPulseClient_H

/*
 * \file EBTestPulseClient.h
 *
 * $Date: 2007/01/27 11:03:02 $
 * $Revision: 1.42 $
 * \author G. Della Ricca
 * \author F. Cossutti
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

class MonitorUserInterface;
class EcalCondDBInterface;
class MonRunIOV;

class EBTestPulseClient : public EBClient {

public:

/// Constructor
EBTestPulseClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBTestPulseClient();

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

string baseHtmlDir_;

vector<int> superModules_;

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

MEContentsProf2DWithinRangeROOT* qtha01_[36];
MEContentsProf2DWithinRangeROOT* qtha02_[36];
MEContentsProf2DWithinRangeROOT* qtha03_[36];

MEContentsProf2DWithinRangeROOT* qtha04_[36];
MEContentsProf2DWithinRangeROOT* qtha05_[36];
MEContentsProf2DWithinRangeROOT* qtha06_[36];
MEContentsProf2DWithinRangeROOT* qtha07_[36];

TProfile2D* hs01_[36];
TProfile2D* hs02_[36];
TProfile2D* hs03_[36];

TH2F* he01_[36];
TH2F* he02_[36];
TH2F* he03_[36];

MonitorElement* meg01_[36];
MonitorElement* meg02_[36];
MonitorElement* meg03_[36];

MonitorElement* meg04_[36];
MonitorElement* meg05_[36];

MonitorElement* mea01_[36];
MonitorElement* mea02_[36];
MonitorElement* mea03_[36];

CollateMonitorElement* me_i01_[36];
CollateMonitorElement* me_i02_[36];
CollateMonitorElement* me_i03_[36];
CollateMonitorElement* me_i04_[36];

TProfile2D* i01_[36];
TProfile2D* i02_[36];
TProfile2D* i03_[36];
TProfile2D* i04_[36];

// Quality check on crystals

float amplitudeThreshold_;
float RMSThreshold_;
float threshold_on_AmplitudeErrorsNumber_;

// Quality check on PNs

float meanThresholdPN_;
float amplitudeThresholdPN_;

};

#endif
