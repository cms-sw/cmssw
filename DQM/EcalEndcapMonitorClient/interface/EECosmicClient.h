#ifndef EECosmicClient_H
#define EECosmicClient_H

/*
 * \file EECosmicClient.h
 *
 * $Date: 2008/04/07 07:24:34 $
 * $Revision: 1.17 $
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
void beginJob(DQMStore* dbe);

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

DQMStore* dbe_;

MonitorElement* meh01_[18];
MonitorElement* meh02_[18];
MonitorElement* meh03_[18];
MonitorElement* meh04_[18];

TProfile2D* h01_[18];
TProfile2D* h02_[18];
TH1F* h03_[18];
TH1F* h04_[18];

};

#endif
