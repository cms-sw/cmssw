#ifndef EEIntegrityClient_H
#define EEIntegrityClient_H

/*
 * \file EEIntegrityClient.h
 *
 * $Date: 2012/04/27 13:46:04 $
 * $Revision: 1.33 $
 * \author G. Della Ricca
 * \author G. Franzoni
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
#ifdef WITH_ECAL_COND_DB
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;
#endif

class EEIntegrityClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EEIntegrityClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEIntegrityClient();

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

#ifdef WITH_ECAL_COND_DB
/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status);
#endif

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

 std::string subfolder_;

bool enableCleanup_;

std::vector<int> superModules_;

DQMStore* dqmStore_;

TH1F* h00_;

TH2F* h01_[18];
TH2F* h02_[18];
TH2F* h03_[18];
TH2F* h04_[18];
TH2F* h05_[18];
TH2F* h06_[18];
TH2F* h07_[18];
TH2F* h08_[18];
TH2F* h09_[18];

MonitorElement* meg01_[18];
MonitorElement* meg02_[18];

TH2F* h_[18];
TH2F* hmem_[18];

// Quality criteria for data integrity

float threshCry_;

const static int chNum [5][5];

};

#endif
