#ifndef EBTimingClient_H
#define EBTimingClient_H

/*
 * \file EBTimingClient.h
 *
 * $Date: 2008/06/25 15:08:17 $
 * $Revision: 1.28 $
 * \author G. Della Ricca
 *
*/

#include <vector>
#include <string>

#include "TROOT.h"
#include "TProfile2D.h"
#include "TH1F.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class MonitorElement;
class DQMStore;
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EBTimingClient : public EBClient {

friend class EBSummaryClient;

public:

/// Constructor
EBTimingClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBTimingClient();

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

/// SoftReset
void softReset(bool flag);

/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag);

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

MonitorElement* meh01_[36];
MonitorElement* meh02_[36];

TProfile2D* h01_[36];
TH2F* h02_[36];

MonitorElement* meg01_[36];

MonitorElement* mea01_[36];

MonitorElement* mep01_[36];

MonitorElement* mer01_[36];

// Quality check on crystals, one per each gain

float expectedMean_;
float discrepancyMean_;
float RMSThreshold_;

};

#endif
