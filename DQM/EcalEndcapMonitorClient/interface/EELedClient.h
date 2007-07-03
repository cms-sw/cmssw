#ifndef EELedClient_H
#define EELedClient_H

/*
 * \file EELedClient.h
 *
 * $Date: 2007/06/11 19:07:31 $
 * $Revision: 1.3 $
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

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class EELedClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EELedClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EELedClient();

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

CollateMonitorElement* me_h01_[18];
CollateMonitorElement* me_h02_[18];

CollateMonitorElement* me_h09_[18];

CollateMonitorElement* me_h13_[18];
CollateMonitorElement* me_h14_[18];

CollateMonitorElement* me_h21_[18];

CollateMonitorElement* me_hs01_[18];

CollateMonitorElement* me_hs05_[18];

TProfile2D* h01_[18];
TProfile2D* h02_[18];

TProfile2D* h09_[18];

TProfile2D* h13_[18];
TProfile2D* h14_[18];

TProfile2D* h21_[18];

MEContentsProf2DWithinRangeROOT* qth01_[18];

MEContentsProf2DWithinRangeROOT* qth05_[18];

MEContentsProf2DWithinRangeROOT* qth09_[18];

MEContentsProf2DWithinRangeROOT* qth13_[18];

MEContentsProf2DWithinRangeROOT* qth17_[18];

MEContentsProf2DWithinRangeROOT* qth21_[18];

TProfile2D* hs01_[18];

TProfile2D* hs05_[18];

MonitorElement* meg01_[18];

MonitorElement* meg05_[18];

MonitorElement* meg09_[18];

MonitorElement* mea01_[18];

MonitorElement* mea05_[18];

MonitorElement* met01_[18];

MonitorElement* met05_[18];

MonitorElement* metav01_[18];

MonitorElement* metav05_[18];

MonitorElement* metrms01_[18];

MonitorElement* metrms05_[18];

MonitorElement* meaopn01_[18];

MonitorElement* meaopn05_[18];

MonitorElement* mepnprms01_[18];

MonitorElement* mepnprms05_[18];

CollateMonitorElement* me_i01_[18];

CollateMonitorElement* me_i05_[18];

CollateMonitorElement* me_i09_[18];

CollateMonitorElement* me_i13_[18];

TProfile2D* i01_[18];

TProfile2D* i05_[18];

TProfile2D* i09_[18];

TProfile2D* i13_[18];

// Quality check on crystals

float percentVariation_;

// Quality check on PNs

float amplitudeThresholdPnG01_;
float amplitudeThresholdPnG16_;
float pedPnExpectedMean_[2];
float pedPnDiscrepancyMean_[2];
float pedPnRMSThreshold_[2];

};

#endif
