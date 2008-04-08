#ifndef EBOccupancyClient_H
#define EBOccupancyClient_H

/*
 * \file EBOccupancyClient.h
 *
 * $Date: 2008/04/07 08:44:19 $
 * $Revision: 1.8 $
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

class EBOccupancyClient : public EBClient {

friend class EBSummaryClient;

public:

/// Constructor
EBOccupancyClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBOccupancyClient();

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

/// HtmlOutput
void htmlOutput(int run, std::string& htmlDir, std::string& htmlName);

/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

private:

int ievt_;
int jevt_;

bool cloneME_;

bool verbose_;
bool debug_;

bool enableCleanup_;

std::vector<int> superModules_;

DQMStore* dqmStore_;

TH2F* h01_[3];
TH1F* h01ProjEta_[3];
TH1F* h01ProjPhi_[3];

TH2F* h02_[2];
TH1F* h02ProjEta_[2];
TH1F* h02ProjPhi_[2];

};

#endif
