#ifndef EEPedestalClient_H
#define EEPedestalClient_H

/*
 * \file EEPedestalClient.h
 *
 * $Date: 2007/08/17 09:05:11 $
 * $Revision: 1.5 $
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
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class EEPedestalClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EEPedestalClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEPedestalClient();

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe(void);
void subscribeNew(void);
void unsubscribe(void);

/// softReset
void softReset(void);

/// Analyze
void analyze(void);

// BeginJob
void beginJob(MonitorUserInterface* mui);

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
void htmlOutput(int run, string htmlDir, string htmlName);

// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

private:

int ievt_;
int jevt_;

bool cloneME_;
bool enableQT_;

bool verbose_;

bool enableMonitorDaemon_;

string prefixME_;

vector<int> superModules_;

MonitorUserInterface* mui_;
DaqMonitorBEInterface* dbe_;

MEContentsProf2DWithinRangeROOT* qth01_[18];
MEContentsProf2DWithinRangeROOT* qth02_[18];
MEContentsProf2DWithinRangeROOT* qth03_[18];

MEContentsProf2DWithinRangeROOT* qth04_[18];
MEContentsProf2DWithinRangeROOT* qth05_[18];

TProfile2D* h01_[18];
TProfile2D* h02_[18];
TProfile2D* h03_[18];

TProfile2D* j01_[18];
TProfile2D* j02_[18];
TProfile2D* j03_[18];

TProfile2D* k01_[18];
TProfile2D* k02_[18];
TProfile2D* k03_[18];

MonitorElement* meg01_[18];
MonitorElement* meg02_[18];
MonitorElement* meg03_[18];

MonitorElement* meg04_[18];
MonitorElement* meg05_[18];

MonitorElement* mep01_[18];
MonitorElement* mep02_[18];
MonitorElement* mep03_[18];

MonitorElement* mer01_[18];
MonitorElement* mer02_[18];
MonitorElement* mer03_[18];
 
MonitorElement* mer04_[18];
MonitorElement* mer05_[18];

MonitorElement* mes01_[18];
MonitorElement* mes02_[18];
MonitorElement* mes03_[18];

MonitorElement* met01_[18];
MonitorElement* met02_[18];
MonitorElement* met03_[18];

TProfile2D* i01_[18];
TProfile2D* i02_[18];

// Quality check on crystals, one per each gain

float expectedMean_[3];
float discrepancyMean_[3];
float RMSThreshold_[3];

// Quality check on PNs
 
float expectedMeanPn_[2];
float discrepancyMeanPn_[2];
float RMSThresholdPn_[2];

MEContentsTH2FWithinRangeROOT* qtg01_[36];
MEContentsTH2FWithinRangeROOT* qtg02_[36];
MEContentsTH2FWithinRangeROOT* qtg03_[36];

MEContentsTH2FWithinRangeROOT* qtg04_[36];
MEContentsTH2FWithinRangeROOT* qtg05_[36];

};

#endif
