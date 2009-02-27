#ifndef EBStatusFlagsClient_H
#define EBStatusFlagsClient_H

/*
 * \file EBStatusFlagsClient.h
 *
 * $Date: 2008/06/25 15:08:17 $
 * $Revision: 1.14 $
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

class EBStatusFlagsClient : public EBClient {

friend class EBSummaryClient;

public:

/// Constructor
EBStatusFlagsClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBStatusFlagsClient();

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

TH2F* h01_[36];

MonitorElement* meh02_[36];

TH1F* h02_[36];

};

#endif
