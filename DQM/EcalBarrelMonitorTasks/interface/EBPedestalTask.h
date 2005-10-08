#ifndef EBPedestalTask_H
#define EBPedestalTask_H

/*
 * \file EBPedestalTask.h
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

class EBPedestalTask: public edm::EDAnalyzer{

friend class EcalBarrelMonitorModule;

public:

/// Constructor
EBPedestalTask(const edm::ParameterSet& ps, TFile* rootFile);

/// Destructor
virtual ~EBPedestalTask();

protected:

/// Analyze digis out of raw data
void analyze(const edm::Event& e, const edm::EventSetup& c);

private:

int ievt;

TProfile2D* hPedMapG01[36];
TProfile2D* hPedMapG06[36];
TProfile2D* hPedMapG12[36];

ofstream logFile;

};

#endif
