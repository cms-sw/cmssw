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
// $Id: ElectronCalibration.cc,v 1.2 2006/09/11 12:44:58 malgeri Exp $
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

      // ----------member data ---------------------------
      std::string rootfile_;
      std::string hitCollection_;
      std::string EBhitCollection;
      std::string digiProducer;
      std::string hitProducer_;
      std::string calibAlgo_;

       MinL3Algorithm algoL3;
       HouseholderDecomposition algoHH;
       CalibrationCluster calibCluster;
       CalibrationCluster::CalibMap ReducedMap;


       vector<int> EventsPerCrystal;
       vector<vector<float> >EventMatrix; 
       vector<float> oldCalibs;
       vector<float> newCalibs;
       vector<float> energyVector;
       vector<float> temp_solution;
       vector<float> solution;

       
       int myMaxHit_save;
       int read_events;
       int used_events;
       int nupdates;
       int checkEnergy;
       int checkOutBoundEnergy;
       unsigned int subsample_;
       unsigned int supermodule_;
       bool makeIteration;
       float BEAM_ENERGY;
       
 
       static const int MIN_IETA = 30;
       static const int MAX_IETA = 40;
       static const int MIN_IPHI = 3;
       static const int MAX_IPHI = 10;

       TH1F* e25;
};
#endif
