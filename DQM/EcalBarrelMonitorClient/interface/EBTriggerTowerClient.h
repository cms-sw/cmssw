#ifndef EBTriggerTowerClient_H
#define EBTriggerTowerClient_H

/*
 * \file EBTriggerTowerClient.h
 *
 * $Date: 2008/03/15 14:50:54 $
 * $Revision: 1.29 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <vector>
#include <string>

#include "TROOT.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TProfile2D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class MonitorElement;
class DQMStore;
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EBTriggerTowerClient : public EBClient {

friend class EBSummaryClient;

public:

/// Constructor
EBTriggerTowerClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBTriggerTowerClient();

/// Analyze
void analyze(void);

/// BeginJob
void beginJob(DQMStore* mui);

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
void htmlOutput(int run, std::string& htmlDir, std::string& htmlName);

/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

private:

void analyze(const char* nameext, const char* folder, bool emulated);

int ievt_;
int jevt_;

bool cloneME_;

bool debug_;

bool enableCleanup_;

std::vector<int> superModules_;

DQMStore* dbe_;

MonitorElement* meh01_[36];
MonitorElement* meh02_[36];
MonitorElement* mei01_[36];
MonitorElement* mei02_[36];
MonitorElement* mej01_[36];
MonitorElement* mej02_[36];

MonitorElement* mel01_[36];
MonitorElement* mem01_[36];
MonitorElement* men01_[36];

TH3F* h01_[36];
TH3F* h02_[36];
TH3F* i01_[36];
TH3F* i02_[36];
TH3F* j01_[36];
TH3F* j02_[36];

TH2F* l01_[36];
TH3F* m01_[36];
TH3F* n01_[36];

//MonitorElement* mek01_[36][68];
//MonitorElement* mek02_[36][68];

//TH1F* k01_[36][68];
//TH1F* k02_[36][68];

MonitorElement* me_h01_[36];
MonitorElement* me_h02_[36];

MonitorElement* me_i01_[36][2];
MonitorElement* me_i02_[36][2];
MonitorElement* me_n01_[36][2];

MonitorElement* me_j01_[36][6];
MonitorElement* me_j02_[36][6];
MonitorElement* me_m01_[36][6];

};

#endif
