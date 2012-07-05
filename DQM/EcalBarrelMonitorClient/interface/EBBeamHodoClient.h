#ifndef EBBeamHodoClient_H
#define EBBeamHodoClient_H

/*
 * \file EBBeamHodoClient.h
 *
 * $Date: 2009/10/28 08:18:20 $
 * $Revision: 1.39 $
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

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class MonitorElement;
class DQMStore;
#ifdef WITH_ECAL_COND_DB
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;
#endif

class EBBeamHodoClient : public EBClient {

public:

/// Constructor
EBBeamHodoClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBBeamHodoClient();

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

TH1F* ho01_[4];
TH1F* hr01_[4];

TH1F* hp01_[2];
TH2F* hp02_;

TH1F* hs01_[2];

TH1F* hq01_[2];

TH1F* ht01_;

TH1F* hc01_[3];

TH1F* hm01_;

TProfile* he01_[2];
TH2F* he02_[2];

TH1F* he03_[3];

};

#endif
