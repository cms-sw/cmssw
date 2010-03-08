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

#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"
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
  virtual void beginJob();
  virtual void beginRun(edm::Run const &, edm::EventSetup const &);

  /// Called at end of job
  virtual void endJob();

  /// Called at each event 
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

  /// 
  void setUp(const edm::EventSetup& setup);

 private:

  // private member functions

  void getKfactors();
  void fillHistos();
  void fillConstantsHistos();

  // private data members


  static const int  kBarlRings  = 85;
  static const int  kBarlWedges = 360;
  static const int  kSides      = 2;

  static const int  kEndcWedgesX = 100;
  static const int  kEndcWedgesY = 100;

  static const int  kEndcEtaRings  = 39;

  static const int  kNMiscalBinsEB = 21;
  static const float  kMiscalRangeEB;

  //  static const int  kNMiscalBinsEE = 81;
  static const int  kNMiscalBinsEE = 41; //VS
  static const float  kMiscalRangeEE;

  // Transverse energy sum arrays
  double etsum_barl_[kBarlRings]  [kBarlWedges] [kSides];
  double etsum_endc_[kEndcWedgesX][kEndcWedgesX][kSides];
  double etsum_endc_uncorr[kEndcWedgesX][kEndcWedgesX][kSides];
  double etsumMean_barl_[kBarlRings];
  double etsumMean_endc_[kEndcEtaRings];

  unsigned int nhits_barl_[kBarlRings][kBarlWedges] [kSides];
  unsigned int nhits_endc_[kEndcWedgesX][kEndcWedgesX][kSides];

  double etsum_barl_miscal_[kNMiscalBinsEB][kBarlRings];
  double etsum_endc_miscal_[kNMiscalBinsEE][kEndcEtaRings];

  double esum_barl_[kBarlRings]  [kBarlWedges] [kSides];
  double esum_endc_[kEndcWedgesX][kEndcWedgesX][kSides];
  double eta_barl_[kBarlRings]  [kBarlWedges] [kSides];
  double eta_endc_[kEndcWedgesX][kEndcWedgesX][kSides];
  double phi_barl_[kBarlRings]  [kBarlWedges] [kSides];
  double phi_endc_[kEndcWedgesX][kEndcWedgesX][kSides];
  double esumMean_barl_[kBarlRings];
  double esumMean_endc_[kEndcEtaRings];


  // crystal geometry information
  double cellEta_[kBarlRings];
  double cellPhiB_[kBarlWedges];

  GlobalPoint cellPos_[kEndcWedgesX][kEndcWedgesY];
  double cellPhi_     [kEndcWedgesX][kEndcWedgesY];
  double cellArea_    [kEndcWedgesX][kEndcWedgesY];
  double meanCellArea_[kEndcEtaRings];
  double etaBoundary_ [kEndcEtaRings+1];
  int endcapRing_     [kEndcWedgesX][kEndcWedgesY];
  int nRing_          [kEndcEtaRings];

  float delta_Eta[kEndcWedgesX][kEndcWedgesY];
  float delta_Phi[kEndcWedgesX][kEndcWedgesY];


  // factors to convert from ET sum deviation to miscalibration
  double k_barl_[kBarlRings];
  double k_endc_[kEndcEtaRings];
  double miscalEB_[kNMiscalBinsEB];
  double miscalEE_[kNMiscalBinsEE]; 

  std::vector<DetId> barrelCells;
  std::vector<DetId> endcapCells;

  // input calibration constants
  double oldCalibs_barl[kBarlRings  ][kBarlWedges][kSides];
  double oldCalibs_endc[kEndcWedgesX][kEndcWedgesY][kSides];

  // new calibration constants
  double newCalibs_barl[kEndcWedgesX][kEndcWedgesX][kSides];
  double newCalibs_endc[kEndcWedgesX][kEndcWedgesX][kSides];

  // calibration constants not multiplied by old ones
  float epsilon_M_barl[kBarlRings][kBarlWedges][kSides];
  float epsilon_M_endc[kEndcWedgesX][kEndcWedgesY][kSides];

  // calibration const not corrected for k
  float rawconst_barl[kBarlRings][kBarlWedges][kSides];
  float rawconst_endc[kEndcWedgesX][kEndcWedgesX][kSides];   

  // informations about good cells
  Bool_t goodCell_barl[kBarlRings][kBarlWedges][kSides];
  Bool_t goodCell_endc[kEndcWedgesX][kEndcWedgesX][kSides];   
  int nBads_barl[kBarlRings];
  int nBads_endc[kEndcEtaRings];

  // steering parameters

  std::string ecalHitsProducer_;
  std::string barrelHits_;
  std::string endcapHits_;
  double eCut_barl_;
  double eCut_endc_;  
  int eventSet_;
  /// threshold in channel status beyond which channel is marked bad
  int statusThreshold_; 

  static const int kMaxEndciPhi = 360;
   
  float phi_endc[kMaxEndciPhi][kEndcEtaRings]; 
  

  bool reiteration_;
  std::string reiterationXMLFileEB_;
  std::string reiterationXMLFileEE_;
  CaloMiscalibMapEcal newCalibs_;
  EcalIntercalibConstants previousCalibs_;
  bool isfirstpass_;
};

#endif
