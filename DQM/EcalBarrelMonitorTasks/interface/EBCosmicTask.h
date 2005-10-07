#ifndef EBCosmicTask_H
#define EBCosmicTask_H

/*
 * \file EBCosmicTask.h
 *
 * $Date: 2005/10/07 08:02:53 $
 * $Revision: 1.1 $
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

class EBCosmicTask: public edm::EDAnalyzer{

friend class EBMonitorModule;

public:

/// Constructor
EBCosmicTask(const edm::ParameterSet& ps, TFile* rootFile);

/// Destructor
virtual ~EBCosmicTask();

protected:

/// Analyze digis out of raw data
void analyze(const edm::Event& e, const edm::EventSetup& c);

private:

int ievt;

TProfile2D* hCutMap[36];

TProfile2D* hSelMap[36];

ofstream logFile;

};

#endif
