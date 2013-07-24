#ifndef EEOccupancyClient_H
#define EEOccupancyClient_H

/*
 * \file EEOccupancyClient.h
 *
 * $Date: 2010/01/25 21:12:25 $
 * $Revision: 1.22 $
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
#ifdef WITH_ECAL_COND_DB
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;
#endif

class EEOccupancyClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EEOccupancyClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEOccupancyClient();

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

bool enableCleanup_;

std::vector<int> superModules_;

DQMStore* dqmStore_;

TH2F* i01_[18];
TProfile2D* i02_[18];

TH2F* h01_[2][3];
TH1F* h01ProjEta_[2][3];
TH1F* h01ProjPhi_[2][3];

TH2F* h02_[2][2];
TH1F* h02ProjEta_[2][2];
TH1F* h02ProjPhi_[2][2];

};

#endif
