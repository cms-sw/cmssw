#ifndef EEPedestalClient_H
#define EEPedestalClient_H

/*
 * \file EEPedestalClient.h
 *
 * $Date: 2010/02/14 20:56:24 $
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

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class MonitorElement;
class DQMStore;
#ifdef WITH_ECAL_COND_DB
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;
#endif

class EEPedestalClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EEPedestalClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEPedestalClient();

/// Analyze
void analyze(void);

// BeginJob
void beginJob(void);

// EndJob
void endJob(void);

// BeginRun
void beginRun(void);

// EndRun
void endRun(void);

/// Setup
void setup(void);

// Cleanup
void cleanup(void);

#ifdef WITH_ECAL_COND_DB
// WriteDB
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
std::vector<int> MGPAGains_;
std::vector<int> MGPAGainsPN_;

DQMStore* dqmStore_;

TProfile2D* h01_[18];
TProfile2D* h02_[18];
TProfile2D* h03_[18];

TProfile2D* j01_[18];
TProfile2D* j02_[18];
TProfile2D* j03_[18];

TProfile2D* k01_[18];
TProfile2D* k02_[18];
TProfile2D* k03_[18];

MonitorElement* meg01_[18];
MonitorElement* meg02_[18];
MonitorElement* meg03_[18];

MonitorElement* meg04_[18];
MonitorElement* meg05_[18];

MonitorElement* mep01_[18];
MonitorElement* mep02_[18];
MonitorElement* mep03_[18];

MonitorElement* mer01_[18];
MonitorElement* mer02_[18];
MonitorElement* mer03_[18];
 
MonitorElement* mer04_[18];
MonitorElement* mer05_[18];

MonitorElement* mes01_[18];
MonitorElement* mes02_[18];
MonitorElement* mes03_[18];

MonitorElement* met01_[18];
MonitorElement* met02_[18];
MonitorElement* met03_[18];

TProfile* i01_[18];
TProfile* i02_[18];

// Quality check on crystals, one per each gain

float expectedMean_[3];
float discrepancyMean_[3];
float RMSThreshold_[3];
float RMSThresholdInner_[3];

// Quality check on PNs
 
float expectedMeanPn_[2];
float discrepancyMeanPn_[2];
float RMSThresholdPn_[2];

};

#endif
