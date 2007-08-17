#ifndef EEClusterClient_H
#define EEClusterClient_H

/*
 * \file EEClusterClient.h
 *
 * $Date: 2007/08/09 14:36:54 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 * \author F. Cossutti
 * \author E. Di Marco
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
#include "DQMServices/Core/interface/CollateMonitorElement.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class EEClusterClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EEClusterClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEClusterClient();

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
DaqMonitorBEInterface* dbe_;

CollateMonitorElement* me_allEEBasic_[3];
CollateMonitorElement* me_eneEEBasic_[2];
CollateMonitorElement* me_numEEBasic_[2];
CollateMonitorElement* me_enePolarEEBasic_[2];
CollateMonitorElement* me_numPolarEEBasic_[2];

CollateMonitorElement* me_allEE_[3];
CollateMonitorElement* me_eneEE_[2];
CollateMonitorElement* me_numEE_[2];
CollateMonitorElement* me_enePolarEE_[2];
CollateMonitorElement* me_numPolarEE_[2];

CollateMonitorElement* me_s_;

TH1F* allEEBasic_[3];
TProfile2D* eneEEBasic_[2];
TH2F* numEEBasic_[2];
TProfile2D* enePolarEEBasic_[2];
TH2F* numPolarEEBasic_[2];

TH1F* allEE_[3];
TProfile2D* eneEE_[2];
TH2F* numEE_[2];
TProfile2D* enePolarEE_[2];
TH2F* numPolarEE_[2];

TH1F* s_;


};

#endif
