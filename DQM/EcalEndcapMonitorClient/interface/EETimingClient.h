#ifndef EETimingClient_H
#define EETimingClient_H

/*
 * \file EETimingClient.h
 *
 * $Date: 2007/11/08 15:43:52 $
 * $Revision: 1.6 $
 * \author G. Della Ricca
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

class EETimingClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EETimingClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EETimingClient();

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

vector<int> superModules_;

MonitorUserInterface* mui_;
DaqMonitorBEInterface* dbe_;

MonitorElement* meh01_[18];

TProfile2D* h01_[18];

MonitorElement* meg01_[18];

MonitorElement* mea01_[18];

MonitorElement* mep01_[18];

MonitorElement* mer01_[18];

// Quality check on crystals, one per each gain

float expectedMean_;
float discrepancyMean_;
float RMSThreshold_;

};

#endif
