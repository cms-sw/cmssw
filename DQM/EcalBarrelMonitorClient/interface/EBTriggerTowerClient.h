#ifndef EBTriggerTowerClient_H
#define EBTriggerTowerClient_H

/*
 * \file EBTriggerTowerClient.h
 *
 * $Date: 2006/12/15 09:44:49 $
 * $Revision: 1.7 $
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
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"

#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

class MonitorUserInterface;
class EcalCondDBInterface;
class MonRunIOV;

class EBTriggerTowerClient : public EBClient {

public:

/// Constructor
EBTriggerTowerClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBTriggerTowerClient();

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

CollateMonitorElement* me_h01_[36];

MonitorElement* meh01_[36];

TProfile2D* h01_[36];

CollateMonitorElement* me_i01_[36];

MonitorElement* mei01_[36];

TH3F* i01_[36];

CollateMonitorElement* me_j01_[36];

MonitorElement* mej01_[36];

TH3F* j01_[36];

CollateMonitorElement* me_k01_[36][68];
CollateMonitorElement* me_k02_[36][68];

MonitorElement* mek01_[36][68];
MonitorElement* mek02_[36][68];

TH1F* k01_[36][68];
TH1F* k02_[36][68];

};

#endif
