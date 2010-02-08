#ifndef EETriggerTowerClient_H
#define EETriggerTowerClient_H

/*
 * \file EETriggerTowerClient.h
 *
 * $Date: 2009/08/27 15:41:03 $
 * $Revision: 1.33 $
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
void beginJob(void);

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

/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

private:

int ievt_;
int jevt_;

bool cloneME_;

bool verbose_;
bool debug_;

std::string prefixME_;

bool enableCleanup_;

std::vector<int> superModules_;

DQMStore* dqmStore_;

MonitorElement* mel01_[18];
MonitorElement* meo01_[18];

TH2F* l01_[18];
TH3F* o01_[18];

MonitorElement* me_o01_[18];
MonitorElement* me_o02_[18];

};

#endif
