#ifndef EcalBarrelMonitorModule_H
#define EcalBarrelMonitorModule_H

/*
 * \file EcalBarrelMonitorModule.h
 *
 * $Date: 2005/10/07 08:47:46 $
 * $Revision: 1.2 $
 *
*/

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBPedestalTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBTestPulseTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBLaserTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBCosmicTask.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBHtmlTask.h>

#include "TROOT.h"
#include "TFile.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace std;

class EcalBarrelMonitorModule: public edm::EDAnalyzer{

public:

/// Constructor
EcalBarrelMonitorModule(const edm::ParameterSet& ps);

/// Destructor
virtual ~EcalBarrelMonitorModule();

protected:

/// Analyze digis out of raw data
void analyze(const edm::Event& e, const edm::EventSetup& c);

private:

int ievt;

TFile* rootFile;

TH1F* hEbarrel;

EBPedestalTask* pedestal_task;
EBTestPulseTask* testpulse_task;
EBLaserTask* laser_task;
EBCosmicTask* cosmic_task;

EBHtmlTask* html_task;

ofstream logFile;

};

DEFINE_FWK_MODULE(EcalBarrelMonitorModule)

#endif
