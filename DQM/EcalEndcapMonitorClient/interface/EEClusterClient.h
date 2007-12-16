#ifndef EEClusterClient_H
#define EEClusterClient_H

/*
 * \file EEClusterClient.h
 *
 * $Date: 2007/12/15 11:34:30 $
 * $Revision: 1.11 $
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
class MonitorUserInterface;
class DaqMonitorBEInterface;
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EEClusterClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EEClusterClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEClusterClient();

/// softReset
void softReset(void);

/// Analyze
void analyze(void);

/// BeginJob
void beginJob(MonitorUserInterface* mui);

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
void htmlOutput(int run, std::string htmlDir, std::string htmlName);

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

bool enableMonitorDaemon_;

std::string prefixME_;

std::vector<int> superModules_;

MonitorUserInterface* mui_;
DaqMonitorBEInterface* dbe_;

TH1F* h01_[3];
TProfile2D* h04_[3][2];
TProfile* h02ProjR_[3][2];
TProfile* h02ProjPhi_[3][2];
TH2F* h03_[2];
TH1F* h03ProjR_[2];
TH1F* h03ProjPhi_[2];
TH1F* i01_[3]; 
TH1F* s01_[3];

};

#endif
