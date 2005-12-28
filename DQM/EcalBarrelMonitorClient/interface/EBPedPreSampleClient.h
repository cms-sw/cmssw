#ifndef EBPedPreSampleClient_H
#define EBPedPreSampleClient_H

/*
 * \file EBPedPreSampleClient.h
 *
 * $Date: 2005/12/26 13:14:24 $
 * $Revision: 1.15 $
 * \author G. Della Ricca
 * \author F. Cossutti
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

//#include "CalibCalorimetry/EcalDBInterface/interface/MonPedPreSampleDat.h"

#include "TROOT.h"
#include "TStyle.h"
#include "TPaveStats.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace std;

class EBPedPreSampleClient: public edm::EDAnalyzer{

friend class EcalBarrelMonitorClient;

public:

/// Constructor
EBPedPreSampleClient(const edm::ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
virtual ~EBPedPreSampleClient();

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
void writeDb(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag);

private:

int ievt_;
int jevt_;

bool collateSources_;

bool verbose_;

MonitorUserInterface* mui_;

CollateMonitorElement* me_h03_[36];

TProfile2D* h03_[36];

TH2F* g03_[36];

TH1F* p03_[36];

TH1F* r03_[36];

// Quality check on crystals, one per each gain

float expectedMean_;
float discrepancyMean_;
float RMSThreshold_; 

};

#endif
