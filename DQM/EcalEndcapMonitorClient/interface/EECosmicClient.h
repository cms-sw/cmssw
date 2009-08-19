#ifndef EECosmicClient_H
#define EECosmicClient_H

/*
 * \file EECosmicClient.h
 *
 * $Date: 2009/02/27 13:54:07 $
 * $Revision: 1.25 $
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

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class MonitorElement;
class DQMStore;
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EECosmicClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EECosmicClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EECosmicClient();

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

MonitorElement* meh01_[18];
MonitorElement* meh02_[18];
MonitorElement* meh03_[18];

TProfile2D* h01_[18];
TH1F* h02_[18];
TH1F* h03_[18];

};

#endif
