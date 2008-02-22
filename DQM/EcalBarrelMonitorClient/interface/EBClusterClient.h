#ifndef EBClusterClient_H
#define EBClusterClient_H

/*
 * \file EBClusterClient.h
 *
 * $Date: 2007/03/26 17:35:04 $
 * $Revision: 1.10 $
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
#include "DQMServices/Core/interface/CollateMonitorElement.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class EBClusterClient : public EBClient {

friend class EBSummaryClient;

public:

/// Constructor
EBClusterClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBClusterClient();

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

bool collateSources_;
bool cloneME_;
bool enableQT_;

bool verbose_;

bool enableMonitorDaemon_;

string prefixME_;

vector<int> superModules_;

MonitorUserInterface* mui_;

CollateMonitorElement* me_h01_[3];
CollateMonitorElement* me_h02_[2];
CollateMonitorElement* me_h03_;
CollateMonitorElement* me_h04_;

CollateMonitorElement* me_i01_[3];
CollateMonitorElement* me_i02_[2];
CollateMonitorElement* me_i03_;
CollateMonitorElement* me_i04_;

CollateMonitorElement* me_s01_[2];

TH1F* h01_[3];
TProfile2D* h02_[2];
TH2F* h03_;
TProfile2D* h04_;

TH1F* i01_[3];
TProfile2D* i02_[2];
TH2F* i03_;
TProfile2D* i04_;

TH1F* s01_[2];

};

#endif
