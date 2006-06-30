#ifndef EBBeamHodoClient_H
#define EBBeamHodoClient_H

/*
 * \file EBBeamHodoClient.h
 *
 * $Date: 2006/06/29 22:03:24 $
 * $Revision: 1.5 $
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
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"

#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

using namespace cms;
using namespace edm;
using namespace std;

class EBBeamHodoClient : public EBClient {

public:

/// Constructor
EBBeamHodoClient(const ParameterSet& ps);

/// Destructor
virtual ~EBBeamHodoClient();

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe(void);
void subscribeNew(void);
void unsubscribe(void);

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
void writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov, int ism);

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

TH1F* ho01_[4];
TH1F* hr01_[4];

TH1F* hp01_[2];
TH2F* hp02_;

TH1F* hs01_[2];

TH1F* hq01_[2];

TH1F* ht01_;

TH1F* hc01_[3];

TProfile* he01_[2];
TH2F* he02_[2];

TH1F* he03_[3];

};

#endif
