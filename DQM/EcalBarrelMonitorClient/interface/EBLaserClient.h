#ifndef EBLaserClient_H
#define EBLaserClient_H

/*
 * \file EBLaserClient.h
 *
 * $Date: 2006/01/02 09:18:01 $
 * $Revision: 1.18 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "CalibCalorimetry/EcalDBInterface/interface/EcalCondDBInterface.h"
#include "CalibCalorimetry/EcalDBInterface/interface/RunTag.h"
#include "CalibCalorimetry/EcalDBInterface/interface/RunIOV.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonRunIOV.h"

#include "CalibCalorimetry/EcalDBInterface/interface/MonLaserBlueDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonLaserGreenDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonLaserIRedDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonLaserRedDat.h"

#include "CalibCalorimetry/EcalDBInterface/interface/MonPNBlueDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonPNGreenDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonPNIRedDat.h"
#include "CalibCalorimetry/EcalDBInterface/interface/MonPNRedDat.h"

#include "TROOT.h"
#include "TStyle.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace std;

class EBLaserClient: public edm::EDAnalyzer{

friend class EcalBarrelMonitorClient;

public:

/// Constructor
EBLaserClient(const edm::ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
virtual ~EBLaserClient();

protected:

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe();
void subscribeNew();
void unsubscribe();

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(const edm::EventSetup& c);

/// EndJob
void endJob(void);

/// BeginRun
void beginRun(const edm::EventSetup& c);

/// EndRun
void endRun(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

/// HtmlOutput
void htmlOutput(int run, string htmlDir, string htmlName);

/// WriteDB
void writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov);

private:

int ievt_;
int jevt_;

bool collateSources_;

bool verbose_;

MonitorUserInterface* mui_;

CollateMonitorElement* me_h01_[36];
CollateMonitorElement* me_h02_[36];
CollateMonitorElement* me_h03_[36];
CollateMonitorElement* me_h04_[36];
CollateMonitorElement* me_h05_[36];
CollateMonitorElement* me_h06_[36];
CollateMonitorElement* me_h07_[36];
CollateMonitorElement* me_h08_[36];

TProfile2D* h01_[36];
TProfile2D* h02_[36];
TProfile2D* h03_[36];
TProfile2D* h04_[36];
TProfile2D* h05_[36];
TProfile2D* h06_[36];
TProfile2D* h07_[36];
TProfile2D* h08_[36];

TH2F* g01_[36];
TH2F* g02_[36];
TH2F* g03_[36];
TH2F* g04_[36];

TH1F* a01_[36];
TH1F* a02_[36];
TH1F* a03_[36];
TH1F* a04_[36];

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
