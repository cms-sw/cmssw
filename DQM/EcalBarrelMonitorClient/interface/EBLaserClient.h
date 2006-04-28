#ifndef EBLaserClient_H
#define EBLaserClient_H

/*
 * \file EBLaserClient.h
 *
 * $Date: 2006/03/05 09:50:40 $
 * $Revision: 1.25 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

#include "OnlineDB/EcalCondDB/interface/MonLaserBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserGreenDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserIRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserRedDat.h"

#include "OnlineDB/EcalCondDB/interface/MonPNBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNGreenDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNIRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNRedDat.h"

#include "TROOT.h"
#include "TStyle.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace edm;
using namespace std;

class EBLaserClient{

public:

/// Constructor
EBLaserClient(const ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
virtual ~EBLaserClient();

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
void htmlOutput(int run, int jsm, string htmlDir, string htmlName);

/// WriteDB
void writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov);

private:

int ievt_;
int jevt_;

bool collateSources_;
bool cloneME_;

bool verbose_;

bool enableMonitorDaemon_;

MonitorUserInterface* mui_;

CollateMonitorElement* me_h01_[36];
CollateMonitorElement* me_h02_[36];
CollateMonitorElement* me_h03_[36];
CollateMonitorElement* me_h04_[36];
CollateMonitorElement* me_h05_[36];
CollateMonitorElement* me_h06_[36];
CollateMonitorElement* me_h07_[36];
CollateMonitorElement* me_h08_[36];

CollateMonitorElement* me_h09_[36];
CollateMonitorElement* me_h10_[36];
CollateMonitorElement* me_h11_[36];
CollateMonitorElement* me_h12_[36];

TProfile2D* h01_[36];
TProfile2D* h02_[36];
TProfile2D* h03_[36];
TProfile2D* h04_[36];
TProfile2D* h05_[36];
TProfile2D* h06_[36];
TProfile2D* h07_[36];
TProfile2D* h08_[36];

TProfile2D* h09_[36];
TProfile2D* h10_[36];
TProfile2D* h11_[36];
TProfile2D* h12_[36];

TH2F* g01_[36];
TH2F* g02_[36];
TH2F* g03_[36];
TH2F* g04_[36];

TH1F* a01_[36];
TH1F* a02_[36];
TH1F* a03_[36];
TH1F* a04_[36];

TH1F* t01_[36];
TH1F* t02_[36];
TH1F* t03_[36];
TH1F* t04_[36];

TH1F* aopn01_[36];
TH1F* aopn02_[36];
TH1F* aopn03_[36];
TH1F* aopn04_[36];

CollateMonitorElement* me_i01_[36];
CollateMonitorElement* me_i02_[36];
CollateMonitorElement* me_i03_[36];
CollateMonitorElement* me_i04_[36];
CollateMonitorElement* me_i05_[36];
CollateMonitorElement* me_i06_[36];
CollateMonitorElement* me_i07_[36];
CollateMonitorElement* me_i08_[36];

TProfile2D* i01_[36];
TProfile2D* i02_[36];
TProfile2D* i03_[36];
TProfile2D* i04_[36];
TProfile2D* i05_[36];
TProfile2D* i06_[36];
TProfile2D* i07_[36];
TProfile2D* i08_[36];

CollateMonitorElement* me_j01_[36];
CollateMonitorElement* me_j02_[36];
CollateMonitorElement* me_j03_[36];
CollateMonitorElement* me_j04_[36];
CollateMonitorElement* me_j05_[36];
CollateMonitorElement* me_j06_[36];
CollateMonitorElement* me_j07_[36];
CollateMonitorElement* me_j08_[36];

TProfile2D* j01_[36];
TProfile2D* j02_[36];
TProfile2D* j03_[36];
TProfile2D* j04_[36];
TProfile2D* j05_[36];
TProfile2D* j06_[36];
TProfile2D* j07_[36];
TProfile2D* j08_[36];

// Quality check on crystals

float percentVariation_;

};

#endif
