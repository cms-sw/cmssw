#ifndef EBBeamCaloClient_H
#define EBBeamCaloClient_H

/*
 * \file EBBeamCaloClient.h
 *
 * $Date: 2006/06/29 22:03:24 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 * \author A. Ghezzi
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

class EBBeamCaloClient : public EBClient {

public:

/// Constructor
EBBeamCaloClient(const ParameterSet& ps);

/// Destructor
virtual ~EBBeamCaloClient();

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
 const static int cryInArray_ = 9;

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

//specific task me

TH1D* meBBCaloGains_[cryInArray_];
TH1D* meBBCaloEne1_;

TH1D* meBBCaloGainsMoving_[cryInArray_];
TH1D* meBBCaloEne1Moving_;


TH1D* meBBCaloAllNeededCry_;

TH1D* meBBCaloE3x3_;

TH1D* meBBCaloE3x3Moving_;

TH2D* meBBCaloCryOnBeam_;

TH2D* meBBCaloMaxEneCry_;


TH1D* meEBBCaloReadCryErrors_;

TH1D* meEBBCaloE1vsCry_;

TH1D* meEBBCaloE3x3vsCry_;

MonitorElement* meEBBCaloRedGreen_;
};

#endif
