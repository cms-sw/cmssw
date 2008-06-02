#ifndef EEPedestalOnlineClient_H
#define EEPedestalOnlineClient_H

/*
 * \file EEPedestalOnlineClient.h
 *
 * $Date: 2007/05/12 09:39:05 $
 * $Revision: 1.2 $
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

class EEPedestalOnlineClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EEPedestalOnlineClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEPedestalOnlineClient();

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

CollateMonitorElement* me_h03_[18];

MEContentsProf2DWithinRangeROOT* qth03_[18];

MonitorElement* meh03_[18];

TProfile2D* h03_[18];

MonitorElement* meg03_[18];

MonitorElement* mep03_[18];

MonitorElement* mer03_[18];

// Quality check on crystals, one per each gain

float expectedMean_;
float discrepancyMean_;
float RMSThreshold_;

MEContentsTH2FWithinRangeROOT* qtg03_[36];

};

#endif
