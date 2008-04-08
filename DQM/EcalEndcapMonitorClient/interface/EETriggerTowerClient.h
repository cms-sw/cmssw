#ifndef EETriggerTowerClient_H
#define EETriggerTowerClient_H

/*
 * \file EETriggerTowerClient.h
 *
 * $Date: 2008/04/07 08:44:20 $
 * $Revision: 1.21 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <vector>
#include <string>

#include "TROOT.h"
#include "TProfile2D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class MonitorElement;
class DQMStore;
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EETriggerTowerClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EETriggerTowerClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EETriggerTowerClient();

/// Analyze
void analyze(void);

/// BeginJob
void beginJob(DQMStore* dqmStore);

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

bool verbose_;
bool debug_;

bool enableCleanup_;

std::vector<int> superModules_;

DQMStore* dqmStore_;

MonitorElement* meh01_[18];
MonitorElement* meh02_[18];
MonitorElement* mei01_[18];
MonitorElement* mei02_[18];
MonitorElement* mej01_[18];
MonitorElement* mej02_[18];

MonitorElement* mel01_[18];
MonitorElement* mem01_[18];
MonitorElement* men01_[18];

TH3F* h01_[18];
TH3F* h02_[18];
TH3F* i01_[18];
TH3F* i02_[18];
TH3F* j01_[18];
TH3F* j02_[18];

TH2F* l01_[18];
TH3F* m01_[18];
TH3F* n01_[18];

//MonitorElement* mek01_[18][34];
//MonitorElement* mek02_[18][34];

//TH1F* k01_[18][34];
//TH1F* k02_[18][34];

MonitorElement* me_h01_[18];
MonitorElement* me_h02_[18];

MonitorElement* me_i01_[18][2];
MonitorElement* me_i02_[18][2];
MonitorElement* me_n01_[18][2];

MonitorElement* me_j01_[18][6];
MonitorElement* me_j02_[18][6];
MonitorElement* me_m01_[18][6];

};

#endif
