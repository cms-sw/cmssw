#ifndef EBTestPulseClient_H
#define EBTestPulseClient_H

/*
 * \file EBTestPulseClient.h
 *
 * $Date: 2007/11/13 13:20:49 $
 * $Revision: 1.58 $
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

class MonitorElement;
class MonitorUserInterface;
class DaqMonitorBEInterface;
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EBTestPulseClient : public EBClient {

friend class EBSummaryClient;

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
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

private:

int ievt_;
int jevt_;

bool cloneME_;

bool verbose_;

bool enableMonitorDaemon_;

string prefixME_;

string baseHtmlDir_;

vector<int> superModules_;

MonitorUserInterface* mui_;
DaqMonitorBEInterface* dbe_;

TProfile2D* ha01_[36];
TProfile2D* ha02_[36];
TProfile2D* ha03_[36];

TProfile2D* hs01_[36];
TProfile2D* hs02_[36];
TProfile2D* hs03_[36];

MonitorElement* meg01_[36];
MonitorElement* meg02_[36];
MonitorElement* meg03_[36];

MonitorElement* meg04_[36];
MonitorElement* meg05_[36];

MonitorElement* mea01_[36];
MonitorElement* mea02_[36];
MonitorElement* mea03_[36];

MonitorElement* mer04_[36];
MonitorElement* mer05_[36];

MonitorElement* me_hs01_[36];
MonitorElement* me_hs02_[36];
MonitorElement* me_hs03_[36];

TProfile* i01_[36];
TProfile* i02_[36];
TProfile* i03_[36];
TProfile* i04_[36];

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
