#ifndef EETestPulseClient_H
#define EETestPulseClient_H

/*
 * \file EETestPulseClient.h
 *
 * $Date: 2008/02/15 13:56:32 $
 * $Revision: 1.18 $
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

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class MonitorElement;
class MonitorUserInterface;
class DaqMonitorBEInterface;
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EETestPulseClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EETestPulseClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EETestPulseClient();

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
void htmlOutput(int run, std::string htmlDir, std::string htmlName);

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

bool enableCleanup_;

std::string prefixME_;

std::vector<int> superModules_;

MonitorUserInterface* mui_;
DaqMonitorBEInterface* dbe_;

TProfile2D* ha01_[18];
TProfile2D* ha02_[18];
TProfile2D* ha03_[18];

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

MonitorElement* me_hs01_[18];
MonitorElement* me_hs02_[18];
MonitorElement* me_hs03_[18];

TProfile* i01_[18];
TProfile* i02_[18];
TProfile* i03_[18];
TProfile* i04_[18];

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
