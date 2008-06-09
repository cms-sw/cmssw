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
#include "FWCore/Framework/interface/EventProcessor.h"
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

  // transverse energy sum arrays
  double etsum_barl_[85][360][2];
  double etsum_endc_[100][100][2];
  double etsumMean_barl_[85];
  double etsumMean_endc_[39];
  double etsum_barl_miscal_[21][85];
  double etsum_endc_miscal_[21][39];

  // crystal geometry information
  double cellEta_[85];
  GlobalPoint cellPos_[100][100];
  double cellPhi_[100][100];
  double cellArea_[100][100];
  double meanCellArea_[39];
  double etaBoundary_[40];
  int endcapRing_[100][100];
  int nRing_[39];

  // factors to convert from ET sum deviation to miscalibration
  double k_barl_[85];
  double k_endc_[39];
  double miscal_[21];
 
  std::vector<DetId> barrelCells;
  std::vector<DetId> endcapCells;

  // input calibration constants
  double oldCalibs_barl[85][360][2];
  double oldCalibs_endc[100][360][2];

  // steering parameters

  edm::ParameterSet theParameterSet;

  unsigned int theMaxLoops;     // Number of loops to loop
  std::string ecalHitsProducer_;
  std::string barrelHits_;
  std::string endcapHits_;
  double eCut_barl_;
  double eCut_endc_;  
  int eventSet_;

   TH1F* etsum_barl_histos_[85]; 
   TH1F* etsum_endc_histos_[39]; 
   TH1F* etsumVsPhi_histos1[39][2]; 
   TH1F* etsumVsPhi_histos[39][2]; 
   TH1F* areaVsPhi_histos[39]; 
   TH1F* etaVsPhi_histos[39]; 
   float phi_endc[300][39]; 
    
   


  TGraph* k_barl_graph_[85];
  TGraph* k_endc_graph_[39];
  TCanvas* k_barl_plot_[85];
  TCanvas* k_endc_plot_[39];
};

#endif
