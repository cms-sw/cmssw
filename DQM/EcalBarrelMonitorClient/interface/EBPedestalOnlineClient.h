#ifndef EBPedestalOnlineClient_H
#define EBPedestalOnlineClient_H

/*
 * \file EBPedestalOnlineClient.h
 *
 * $Date: 2012/04/27 13:45:57 $
 * $Revision: 1.59 $
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

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class MonitorElement;
class DQMStore;
#ifdef WITH_ECAL_COND_DB
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;
#endif

class EBPedestalOnlineClient : public EBClient {

friend class EBSummaryClient;

public:

/// Constructor
EBPedestalOnlineClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBPedestalOnlineClient();

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

TProfile2D* h03_[36];

MonitorElement* meg03_[36];

MonitorElement* mep03_[36];

MonitorElement* mer03_[36];

// Quality check on crystals, one per each gain

float expectedMean_;
float discrepancyMean_;
float RMSThreshold_, RMSThresholdHighEta_;

};

#endif
