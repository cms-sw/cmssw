#ifndef EBPnDiodeClient_H
#define EBPnDiodeClient_H

/*
 * \file EBPnDiodeClient.h
 *
 * $Date: 2005/11/24 09:20:29 $
 * $Revision: 1.1 $
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

#include "CalibCalorimetry/EcalDBInterface/interface/MonPNDat.h"

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

class EBPnDiodeClient: public edm::EDAnalyzer{

friend class EcalBarrelMonitorClient;

public:

/// Constructor
EBPnDiodeClient(const edm::ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
virtual ~EBPnDiodeClient();

protected:

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe();
void subscribeNew();
void unsubscribe();

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

// BeginRun
void beginRun(const edm::EventSetup& c);

// EndRun
void endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag);

// HtmlOutput
virtual void htmlOutput(int run, string htmlDir, string htmlName);

private:

int ievt_;
int jevt_;

MonitorUserInterface* mui_;

TProfile* h01_[36];
TProfile* h02_[36];
TProfile* h03_[36];
TProfile* h04_[36];

};

#endif
