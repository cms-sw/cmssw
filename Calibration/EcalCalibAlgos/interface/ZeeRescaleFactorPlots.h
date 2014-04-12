#ifndef ZEERESCALEFACTORPLOTS_H
#define  ZEERESCALEFACTORPLOTS_H

// -*- C++ -*-
//
// Package:    ElectronCalibration
// Class:      ElectronCalibration
// 
/**\class ElectronCalibration ElectronCalibration.cc Calibration/EcalCalibAlgos/src/ElectronCalibration.cc

 Description: Perform single electron calibration (tested on TB data only).

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo AGOSTINO, Radek Ofierzynski
//         Created:  Tue Jul 18 12:17:01 CEST 2006
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "Calibration/Tools/interface/ZIterativeAlgorithmWithFit.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"


// class declaration
//

class ZeeRescaleFactorPlots {
 
 public:
  ZeeRescaleFactorPlots( char* );
  ~ZeeRescaleFactorPlots();
  
  void writeHistograms( ZIterativeAlgorithmWithFit* );
  
 private:
  
  TFile* file_;
  char* fileName_;
  
};
#endif
