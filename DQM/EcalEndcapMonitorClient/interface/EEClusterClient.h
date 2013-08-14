#ifndef EEClusterClient_H
#define EEClusterClient_H

/*
 * \file EEClusterClient.h
 *
 * $Date: 2010/01/25 21:12:25 $
 * $Revision: 1.31 $
 * \author G. Della Ricca
 * \author F. Cossutti
 * \author E. Di Marco
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

class EEClusterClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EEClusterClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEClusterClient();

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

TH1F* h01_[3];
TProfile2D* h04_[3][2];
TProfile* h02ProjEta_[3][2];
TProfile* h02ProjPhi_[3][2];
TH2F* h03_[2];
TH1F* h03ProjEta_[2];
TH1F* h03ProjPhi_[2];
TH1F* i01_[3]; 
TH1F* s01_[3];

};

#endif
