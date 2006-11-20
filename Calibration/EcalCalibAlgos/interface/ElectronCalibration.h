#ifndef CALIBRATION_ECALCALIBALGOS_ELECTRONCALIBRATION
#define CALIBRATION_ECALCALIBALGOS_ELECTRONCALIBRATION

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
// $Id: ElectronCalibration.h,v 1.3 2006/10/27 14:05:25 lorenzo Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "TFile.h"
#include "TH1.h"


// class decleration
//

class ElectronCalibration : public edm::EDAnalyzer {
   public:
      explicit ElectronCalibration(const edm::ParameterSet&);
      ~ElectronCalibration();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();
   private:

      EBDetId  findMaxHit(edm::Handle<EBRecHitCollection> &);
      EBDetId  findMaxHit2(const std::vector<DetId> & ,const EBRecHitCollection* );

      // ----------member data ---------------------------
      std::string rootfile_;
      edm::InputTag recHitLabel_;
      edm::InputTag electronLabel_;
      edm::InputTag trackLabel_;
      std::string calibAlgo_;
      CalibrationCluster calibCluster;
      CalibrationCluster::CalibMap ReducedMap;
      
      int read_events;
      
      int calibClusterSize;
      int etaMin, etaMax, phiMin, phiMax;
      vector<float> EnergyVector;
      vector<vector<float> > EventMatrix;
      vector<int> MaxCCeta;
      vector<int> MaxCCphi;
      MinL3Algorithm* MyL3Algo1;
      vector<float> solution;
      vector<float> newCalibs;
      vector<float> oldCalibs;
      
      TH1F * e25;
      TH1F * scE;
      TH1F * trP;
      TH1F * EoP;
      TH1F * calibs;
      
};
#endif
