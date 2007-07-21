#ifndef EETestPulseClient_H
#define EETestPulseClient_H

/*
 * \file EETestPulseClient.h
 *
 * $Date: 2007/07/19 11:54:12 $
 * $Revision: 1.4 $
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

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class EETestPulseClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EETestPulseClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EETestPulseClient();

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

CollateMonitorElement* me_ha01_[18];
CollateMonitorElement* me_ha02_[18];
CollateMonitorElement* me_ha03_[18];

CollateMonitorElement* me_hs01_[18];
CollateMonitorElement* me_hs02_[18];
CollateMonitorElement* me_hs03_[18];

TProfile2D* ha01_[18];
TProfile2D* ha02_[18];
TProfile2D* ha03_[18];

MEContentsProf2DWithinRangeROOT* qtha01_[18];
MEContentsProf2DWithinRangeROOT* qtha02_[18];
MEContentsProf2DWithinRangeROOT* qtha03_[18];

MEContentsProf2DWithinRangeROOT* qtha04_[18];
MEContentsProf2DWithinRangeROOT* qtha05_[18];
MEContentsProf2DWithinRangeROOT* qtha06_[18];
MEContentsProf2DWithinRangeROOT* qtha07_[18];

TProfile2D* hs01_[18];
TProfile2D* hs02_[18];
TProfile2D* hs03_[18];

MonitorElement* meg01_[18];
MonitorElement* meg02_[18];
MonitorElement* meg03_[18];

MonitorElement* meg04_[18];
MonitorElement* meg05_[18];

MonitorElement* mea01_[18];
MonitorElement* mea02_[18];
MonitorElement* mea03_[18];

MonitorElement* mer04_[18];
MonitorElement* mer05_[18];

CollateMonitorElement* me_i01_[18];
CollateMonitorElement* me_i02_[18];
CollateMonitorElement* me_i03_[18];
CollateMonitorElement* me_i04_[18];

TProfile2D* i01_[18];
TProfile2D* i02_[18];
TProfile2D* i03_[18];
TProfile2D* i04_[18];

// Quality check on crystals

float percentVariation_;
float RMSThreshold_;

// Quality check on PNs

float amplitudeThresholdPnG01_;
float amplitudeThresholdPnG16_;
float pedPnExpectedMean_[2];
float pedPnDiscrepancyMean_[2];
float pedPnRMSThreshold_[2];

};

#endif
