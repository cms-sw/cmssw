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
#include "Geometry/Vector/interface/GlobalPoint.h"

// Framework
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/ESProducerLooper.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TF1.h"
#include "TGraph.h"
#include "TCanvas.h"

class PhiSymmetryCalibration :  public edm::ESProducerLooper
{

 public:

  /// Constructor
  PhiSymmetryCalibration( const edm::ParameterSet& iConfig );
  
  /// Destructor
  ~PhiSymmetryCalibration();

  /// Dummy implementation (job done in duringLoop)
  virtual void produce(edm::Event&, const edm::EventSetup&) {};

  /// Called at beginning of job
  virtual void beginOfJob(const edm::EventSetup&);

  /// Called at end of job
  virtual void endOfJob();

  /// Called at beginning of loop
  virtual void startingNewLoop( unsigned int iLoop );

  /// Called at end of loop
  virtual Status endOfLoop( const edm::EventSetup&, unsigned int iLoop );

  /// Called at each event 
  virtual Status duringLoop( const edm::Event&, const edm::EventSetup& );

 private:

  // private member functions

  void getCellAreas();
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
  double oldCalibs_endc[85][360][2];

  // steering parameters

  edm::ParameterSet theParameterSet;

  unsigned int theMaxLoops;     // Number of loops to loop
  std::string ecalHitsProducer_;
  std::string barrelHits_;
  std::string endcapHits_;
  double eCut_barl_;
  double eCut_endc_;  

  // root tree
  TFile* theFile;

  TH1F* etsum_barl_histos_[85];
  TH1F* etsum_endc_histos_[39];

  TGraph* k_barl_graph_[85];
  TGraph* k_endc_graph_[39];
  TCanvas* k_barl_plot_[85];
  TCanvas* k_endc_plot_[39];
};

#endif
