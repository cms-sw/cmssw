#ifndef EBTriggerTowerClient_H
#define EBTriggerTowerClient_H

/*
 * \file EBTriggerTowerClient.h
 *
 * $Date: 2009/08/27 15:31:31 $
 * $Revision: 1.43 $
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

MonitorElement* mel01_[36];
MonitorElement* meo01_[36];
TH2F* l01_[36];
TH3F* o01_[36];

MonitorElement* me_o01_[36];
MonitorElement* me_o02_[36];

};

#endif
