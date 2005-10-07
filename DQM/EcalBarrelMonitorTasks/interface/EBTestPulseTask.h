#ifndef EBTestPulseTask_H
#define EBTestPulseTask_H

/*
 * \file EBTestPulseTask.h
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

class EBTestPulseTask: public edm::EDAnalyzer{

friend class EBMonitorModule;

public:

/// Constructor
EBTestPulseTask(const edm::ParameterSet& ps, TFile* rootFile);

/// Destructor
virtual ~EBTestPulseTask();

protected:

/// Analyze digis out of raw data
void analyze(const edm::Event& e, const edm::EventSetup& c);

private:

int ievt;

TProfile2D* hShapeMapG01[36];
TProfile2D* hShapeMapG06[36];
TProfile2D* hShapeMapG12[36];

TProfile2D* hAmplMapG01[36];
TProfile2D* hAmplMapG06[36];
TProfile2D* hAmplMapG12[36];

ofstream logFile;

};

#endif
