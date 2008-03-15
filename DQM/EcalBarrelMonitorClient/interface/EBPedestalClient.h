#ifndef EBPedestalClient_H
#define EBPedestalClient_H

/*
 * \file EBPedestalClient.h
 *
 * $Date: 2008/03/14 14:38:54 $
 * $Revision: 1.67 $
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
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EBPedestalClient : public EBClient {

friend class EBSummaryClient;

public:

/// Constructor
EBPedestalClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBPedestalClient();

/// Analyze
void analyze(void);

// BeginJob
void beginJob(DQMStore* mui);

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

// HtmlOutput
void htmlOutput(int run, std::string& htmlDir, std::string& htmlName);

// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

private:

int ievt_;
int jevt_;

bool cloneME_;

bool verbose_;

bool enableCleanup_;

std::vector<int> superModules_;

DQMStore* dbe_;

TProfile2D* h01_[36];
TProfile2D* h02_[36];
TProfile2D* h03_[36];

TProfile2D* j01_[36];
TProfile2D* j02_[36];
TProfile2D* j03_[36];

TProfile2D* k01_[36];
TProfile2D* k02_[36];
TProfile2D* k03_[36];

MonitorElement* meg01_[36];
MonitorElement* meg02_[36];
MonitorElement* meg03_[36];

MonitorElement* meg04_[36];
MonitorElement* meg05_[36];

MonitorElement* mep01_[36];
MonitorElement* mep02_[36];
MonitorElement* mep03_[36];

MonitorElement* mer01_[36];
MonitorElement* mer02_[36];
MonitorElement* mer03_[36];

MonitorElement* mer04_[36];
MonitorElement* mer05_[36];

MonitorElement* mes01_[36];
MonitorElement* mes02_[36];
MonitorElement* mes03_[36];

MonitorElement* met01_[36];
MonitorElement* met02_[36];
MonitorElement* met03_[36];

TProfile* i01_[36];
TProfile* i02_[36];

// Quality check on crystals, one per each gain

float expectedMean_[3];
float discrepancyMean_[3];
float RMSThreshold_[3];

// Quality check on PNs

float expectedMeanPn_[2];
float discrepancyMeanPn_[2];
float RMSThresholdPn_[2];

};

#endif
