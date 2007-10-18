#ifndef EEClusterClient_H
#define EEClusterClient_H

/*
 * \file EEClusterClient.h
 *
 * $Date: 2007/09/07 22:30:06 $
 * $Revision: 1.6 $
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

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class EEClusterClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EEClusterClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEClusterClient();

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe(void);
void subscribeNew(void);
void unsubscribe(void);

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
void htmlOutput(int run, string htmlDir, string htmlName);

/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

private:

int ievt_;
int jevt_;

bool cloneME_;
bool enableQT_;

bool verbose_;

bool enableMonitorDaemon_;

string prefixME_;

vector<int> superModules_;

MonitorUserInterface* mui_;
DaqMonitorBEInterface* dbe_;

TH1F* hBC1D_[3];
TProfile2D* hProfMap_[3][2];
TProfile* hProfMapProjR_[3][2];
TProfile* hProfMapProjPhi_[3][2];
TH2F* hOccMap_[2];
TH1F* hOccMapProjR_[2];
TH1F* hOccMapProjPhi_[2];
TH1F* hSC1D_[3]; 
TH1F* s01_[3];

};

#endif
