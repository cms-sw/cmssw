#ifndef EBPedestalOnlineClient_H
#define EBPedestalOnlineClient_H

/*
 * \file EBPedestalOnlineClient.h
 *
 * $Date: 2006/06/13 10:08:35 $
 * $Revision: 1.12 $
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

class MonitorUserInterface;
class EcalCondDBInterface;
class MonRunIOV;

class EBPedestalOnlineClient : public EBClient {

public:

/// Constructor
EBPedestalOnlineClient(const ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
virtual ~EBPedestalOnlineClient();

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe(void);
void subscribeNew(void);
void unsubscribe(void);

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

/// HtmlOutput
void htmlOutput(int run, const std::vector<int> & superModules, string htmlDir, string htmlName);

/// WriteDB
void writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov, int ism);

private:

int ievt_;
int jevt_;

bool collateSources_;
bool cloneME_;

bool verbose_;

bool enableMonitorDaemon_;

MonitorUserInterface* mui_;

CollateMonitorElement* me_h03_[36];

MonitorElement* meh03_[36];

TProfile2D* h03_[36];

MonitorElement* meg03_[36];

MEContentsProf2DWithinRangeROOT* qth03_[36];

MonitorElement* mep03_[36];

MonitorElement* mer03_[36];

// Quality check on crystals, one per each gain

float expectedMean_;
float discrepancyMean_;
float RMSThreshold_;

};

#endif
