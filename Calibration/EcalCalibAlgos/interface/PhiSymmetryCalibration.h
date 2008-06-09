#ifndef Calibration_EcalCalibAlgos_PhiSymmetryCalibration_h
#define Calibration_EcalCalibAlgos_PhiSymmetryCalibration_h

//
// Package:    Calibration/EcalCalibAlgos
// Class:      PhiSymmetryCalibration
// 
//
// Description: performs phi-symmetry calibration
//
//
// Original Author:  David Futyan

#include <vector>

// obsolete #include "Geometry/Vector/interface/GlobalPoint.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
// Framework
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TF1.h"
#include "TGraph.h"
#include "TCanvas.h"

class PhiSymmetryCalibration :  public edm::EDAnalyzer
{

 public:

  /// Constructor
  PhiSymmetryCalibration( const edm::ParameterSet& iConfig );
  
  /// Destructor
  ~PhiSymmetryCalibration();

  /// Called at beginning of job
  virtual void beginJob(const edm::EventSetup&);

  /// Called at end of job
  virtual void endJob();

  /// Called at each event 
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

 private:

  // private member functions

  void getKfactors();
  void fillHistos();

  // private data members

  int nevent;


  static const int  kBarlRings  = 85;
  static const int  kBarlWedges = 360;
  static const int  kSides      = 2;

  static const int  kEndcWedgesX = 100;
  static const int  kEndcWedgesY = 100;

  static const int  kEndcEtaRings = 39;
  static const int  kNMiscalBins  = 21;

  // Transverse energy sum arrays
  double etsum_barl_[kBarlRings]  [kBarlWedges] [kSides];
  double etsum_endc_[kEndcWedgesX][kEndcWedgesX][kSides];

  double etsumMean_barl_[kBarlRings];
  double etsumMean_endc_[kEndcEtaRings];

  double etsum_barl_miscal_[kNMiscalBins][kBarlRings];
  double etsum_endc_miscal_[kNMiscalBins][kEndcEtaRings];

  // crystal geometry information
  double cellEta_[kBarlRings];

  GlobalPoint cellPos_[kEndcWedgesX][kEndcWedgesY];
  double cellPhi_     [kEndcWedgesX][kEndcWedgesY];
  double cellArea_    [kEndcWedgesX][kEndcWedgesY];
  double meanCellArea_[kEndcEtaRings];
  double etaBoundary_ [kEndcEtaRings+1];
  int endcapRing_     [kEndcWedgesX][kEndcWedgesY];
  int nRing_          [kEndcEtaRings];

  // factors to convert from ET sum deviation to miscalibration
  double k_barl_[kBarlRings];
  double k_endc_[kEndcEtaRings];
  double miscal_[kNMiscalBins];
 
  std::vector<DetId> barrelCells;
  std::vector<DetId> endcapCells;

  // input calibration constants
  double oldCalibs_barl[kBarlRings  ][kBarlWedges][kSides];
  double oldCalibs_endc[kEndcWedgesX][kEndcWedgesY][kSides];

  // steering parameters

  std::string ecalHitsProducer_;
  std::string barrelHits_;
  std::string endcapHits_;
  double eCut_barl_;
  double eCut_endc_;  
  int eventSet_;

  static const int kMaxEndciPhi = 360;
   
  float phi_endc[kMaxEndciPhi][kEndcEtaRings]; 
  

};

#endif
