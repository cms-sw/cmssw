#ifndef CALIBRATION_ECALCALIBALGOS_ELECTRONCALIBRATIONUNIV
#define CALIBRATION_ECALCALIBALGOS_ELECTRONCALIBRATIONUNIV

// -*- C++ -*-
//
// Package:    ElectronCalibrationUniv
// Class:      ElectronCalibrationUniv
// 
/**\class ElectronCalibrationUniv ElectronCalibrationUniv.cc Calibration/EcalCalibAlgos/src/ElectronCalibrationUniv.cc

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

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Geometry
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

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
#include "TH2.h"

#include "Calibration/Tools/interface/MinL3AlgoUniv.h"

// class decleration
//

class ElectronCalibrationUniv : public edm::EDAnalyzer {
   public:
      explicit ElectronCalibrationUniv(const edm::ParameterSet&);
      ~ElectronCalibrationUniv();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob();
      virtual void beginRun(edm::Run const &, edm::EventSetup const&);
      virtual void endJob();
   private:
      DetId  findMaxHit(const std::vector<DetId> & v1,const EBRecHitCollection *EBhits,const EERecHitCollection *EEhits);
      bool TestEEvalidDetId(int crystal_ix, int crystal_iy, int iz);     


      // ----------member data ---------------------------

      std::string rootfile_;
      edm::InputTag EBrecHitLabel_;
      edm::InputTag EErecHitLabel_;
      edm::InputTag electronLabel_;
      edm::InputTag trackLabel_;
      std::string calibAlgo_;
      std::string miscalibfile_;
      std::string miscalibfileEndCap_;
      int keventweight_;
      double ElePt_;
      int maxeta_;
      int mineta_;
      int maxphi_;
      int minphi_;
      double cut1_;
      double cut2_;
      double cut3_;
      double cutEPCalo1_;
      double cutEPCalo2_;
      double cutEPin1_;
      double cutEPin2_;
      double cutCalo1_;
      double cutCalo2_;
      double cutESeed_;
      int ClusterSize_;
     int elecclass_;
      int theMaxLoops;

      bool FirstIteration;

      int read_events;
      
      int calibClusterSize,keventweight;
      int etaMin, etaMax, phiMin, phiMax;
      std::vector<float> EnergyVector;
      std::vector<float> WeightVector;
      std::vector<std::vector<float> > EventMatrix;
      std::vector<int> MaxCCeta;
      std::vector<int> MaxCCphi;
      std::vector<float> EnergyVectorNoCuts;
      std::vector<float> WeightVectorNoCuts;
      std::vector<std::vector<float> > EventMatrixNoCuts;
      std::vector<int> MaxCCetaNoCuts;
      std::vector<int> MaxCCphiNoCuts;
      MinL3Algorithm* MyL3Algo1;
      MinL3AlgoUniv<DetId>* UnivL3;

      edm::ESHandle<CaloTopology> theCaloTopology;

      std::vector<float> solution;
      std::vector<float> solutionNoCuts;

      //For Universal Algo
      std::vector<std::vector<DetId> > UnivEventIds;
      std::map<DetId,float> Univsolution;
      
      // int eventcrystal[25][25];
      int numevent_;

      TFile * f;

      TH1F * EventsAfterCuts;

      TH1F * e25;
      TH1F * e9;
      TH1F * scE;
      TH1F * trP;
      TH1F * EoP;
      TH1F * EoP_all;
      TH1F * calibs;
      TH1F * calibsEndCapMinus;
      TH1F * calibsEndCapPlus;
      TH1F * e9Overe25;
      TH1F * e25OverScE;
      TH2F * Map;
      TH1F * E25oP;

      TH1F * PinOverPout;
      TH1F * eSeedOverPout;
/*       TH1F * MisCalibs; */
/*       TH1F * RatioCalibs; */
/*       TH1F * DiffCalibs; */
/*       TH1F * RatioCalibsNoCuts; */
/*       TH1F * DiffCalibsNoCuts; */
/*      TH1F * MisCalibsEndCapMinus; */
/*      TH1F * MisCalibsEndCapPlus; */
/*       TH1F * RatioCalibsEndCapMinus; */
/*       TH1F * RatioCalibsEndCapPlus; */
/*       TH1F * DiffCalibsEndCapMinus; */
/*       TH1F * DiffCalibsEndCapPlus; */

      TH1F * e25NoCuts;
      TH1F * e9NoCuts;
      TH1F * scENoCuts;
      TH1F * trPNoCuts;
      TH1F * EoPNoCuts;
      TH1F * calibsNoCuts;
      TH1F * e9Overe25NoCuts;
      TH1F * e25OverScENoCuts;
      TH2F * MapEndCapMinus;
      TH2F * MapEndCapPlus;
      TH1F * E25oPNoCuts;
      TH2F * Map3Dcalib;
      TH2F * Map3DcalibEndCapMinus;
      TH2F * Map3DcalibEndCapPlus;
      TH2F * Map3DcalibNoCuts;
      TH1F * calibinter;
      TH1F * calibinterEndCapMinus;
      TH1F * calibinterEndCapPlus;
      TH1F * calibinterNoCuts;
      HouseholderDecomposition * MyHH;
      TH1F * PinOverPoutNoCuts;
      TH1F * eSeedOverPoutNoCuts;

      TH2F * GeneralMap;
      TH2F * GeneralMapEndCapMinus;
      TH2F * GeneralMapEndCapPlus;
      TH2F * GeneralMapBeforePt;
      TH2F * GeneralMapEndCapMinusBeforePt;
      TH2F * GeneralMapEndCapPlusBeforePt;

      TH2F * MapCor1;
      TH2F * MapCor2;
      TH2F * MapCor3;
      TH2F * MapCor4;
      TH2F * MapCor5;
      TH2F * MapCor6;
      TH2F * MapCor7;
      TH2F * MapCor8;
      TH2F * MapCor9;
      TH2F * MapCor10;
      TH2F * MapCor11;
      //      TH2F * MapCorCalib;

      TH2F * MapCor1NoCuts;
      TH2F * MapCor2NoCuts;
      TH2F * MapCor3NoCuts;
      TH2F * MapCor4NoCuts;
      TH2F * MapCor5NoCuts;
      TH2F * MapCor6NoCuts;
      TH2F * MapCor7NoCuts;
      TH2F * MapCor8NoCuts;
      TH2F * MapCor9NoCuts;
      TH2F * MapCor10NoCuts;
      TH2F * MapCor11NoCuts;
/*       TH2F * MapCorCalibEndCapMinus; */
/*       TH2F * MapCorCalibEndCapPlus; */

      TH2F * MapCor1ESeed;
      TH2F * MapCor2ESeed;
      TH2F * MapCor3ESeed;
      TH2F * MapCor4ESeed;
      TH2F * MapCor5ESeed;
      TH2F * MapCor6ESeed;
      TH2F * MapCor7ESeed;
      TH2F * MapCor8ESeed;
      TH2F * MapCor9ESeed;
      TH2F * MapCor10ESeed;
      TH2F * MapCor11ESeed;

      TH2F * E25oPvsEta;
      TH2F * E25oPvsEtaEndCapMinus;
      TH2F * E25oPvsEtaEndCapPlus;

      TH1F * PinMinPout; 
      TH1F * PinMinPoutNoCuts;

      TH1F * Error1;
      TH1F * Error2;
      TH1F * Error3;
      TH1F * Error1NoCuts;
      TH1F * Error2NoCuts;
      TH1F * Error3NoCuts;

      TH1F * eSeedOverPout2;
      TH1F * eSeedOverPout2NoCuts;
      TH1F * eSeedOverPout2ESeed;

      TH1F * hadOverEm;
      TH1F * hadOverEmNoCuts;
      TH1F * hadOverEmESeed;
};
#endif
