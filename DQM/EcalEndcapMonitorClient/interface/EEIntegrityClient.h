#ifndef EEIntegrityClient_H
#define EEIntegrityClient_H

/*
 * \file EEIntegrityClient.h
 *
 * $Date: 2007/05/12 09:39:05 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 * \author G. Franzoni
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

class EEIntegrityClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EEIntegrityClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEIntegrityClient();

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

bool collateSources_;
bool cloneME_;
bool enableQT_;

bool verbose_;

bool enableMonitorDaemon_;

string prefixME_;

vector<int> superModules_;

MonitorUserInterface* mui_;

CollateMonitorElement* me_h00_;

CollateMonitorElement* me_h01_[18];
CollateMonitorElement* me_h02_[18];
CollateMonitorElement* me_h03_[18];
CollateMonitorElement* me_h04_[18];
CollateMonitorElement* me_h05_[18];
CollateMonitorElement* me_h06_[18];
CollateMonitorElement* me_h07_[18];
CollateMonitorElement* me_h08_[18];
CollateMonitorElement* me_h09_[18];
CollateMonitorElement* me_h10_[18];

TH1F* h00_;

TH2F* h01_[18];
TH2F* h02_[18];
TH2F* h03_[18];
TH2F* h04_[18];
TH2F* h05_[18];
TH2F* h06_[18];
TH2F* h07_[18];
TH2F* h08_[18];
TH2F* h09_[18];
TH2F* h10_[18];

MEContentsTH2FWithinRangeROOT* qth01_[18];
MEContentsTH2FWithinRangeROOT* qth02_[18];
MEContentsTH2FWithinRangeROOT* qth03_[18];
MEContentsTH2FWithinRangeROOT* qth04_[18];
MEContentsTH2FWithinRangeROOT* qth05_[18];
MEContentsTH2FWithinRangeROOT* qth06_[18];
MEContentsTH2FWithinRangeROOT* qth07_[18];
MEContentsTH2FWithinRangeROOT* qth08_[18];
MEContentsTH2FWithinRangeROOT* qth09_[18];
MEContentsTH2FWithinRangeROOT* qth10_[18];

MonitorElement* meg01_[18];
MonitorElement* meg02_[18];

CollateMonitorElement* me_h_[18];
CollateMonitorElement* me_hmem_[18];

TH2F* h_[18];
TH2F* hmem_[18];

// Quality criteria for data integrity

float threshCry_;

const static int chNum [5][5];

MEContentsTH2FWithinRangeROOT* qtg01_[36];
MEContentsTH2FWithinRangeROOT* qtg02_[36];

};

#endif
