#ifndef EBLaserTask_H
#define EBLaserTask_H

/*
 * \file EBLaserTask.h
 *
 * $Date: 2005/10/07 11:15:53 $
 * $Revision: 1.3 $
 * \author G. Della Ricca
 *
*/

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBMonitorUtils.h>

#include "TROOT.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TProfile2D.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace cms;
using namespace std;

class EBLaserTask: public edm::EDAnalyzer{

friend class EcalBarrelMonitorModule;

public:

/// Constructor
EBLaserTask(const edm::ParameterSet& ps, TFile* rootFile);

/// Destructor
virtual ~EBLaserTask();

protected:

/// Analyze digis out of raw data
void analyze(const edm::Event& e, const edm::EventSetup& c);

private:

int ievt;

TProfile2D* hShapeMapL1[36];
TProfile2D* hAmplMapL1[36];

TProfile2D* hShapeMapL2[36];
TProfile2D* hAmplMapL2[36];

ofstream logFile;

};

#endif
