#ifndef EBTimingClient_H
#define EBTimingClient_H

/*
 * \file EBTimingClient.h
 *
 * $Date: 2007/04/29 18:13:43 $
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

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class EBTimingClient : public EBClient {

friend class EBSummaryClient;

public:

/// Constructor
EBTimingClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBTimingClient();

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

CollateMonitorElement* me_h01_[36];

MEContentsProf2DWithinRangeROOT* qth01_[36];

MonitorElement* meh01_[36];

TProfile2D* h01_[36];

MonitorElement* meg01_[36];

MonitorElement* mea01_[36];

MonitorElement* mep01_[36];

MonitorElement* mer01_[36];

// Quality check on crystals, one per each gain

float expectedMean_;
float discrepancyMean_;
float RMSThreshold_;

MEContentsTH2FWithinRangeROOT* qtg01_[36];

};

#endif
