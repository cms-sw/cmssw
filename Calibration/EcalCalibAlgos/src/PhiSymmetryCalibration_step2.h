#include "Calibration/EcalCalibAlgos/interface/EcalGeomPhiSymHelper.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

class TH1F;
class TH2F;

class PhiSymmetryCalibration_step2 :  public edm::EDAnalyzer
{

 public:

  PhiSymmetryCalibration_step2(const edm::ParameterSet& iConfig);
  ~PhiSymmetryCalibration_step2();
  
  void beginJob();
  void endJob();
  
  void analyze( const edm::Event&, const edm::EventSetup& );

  void fillHistos();
  void fillConstantsHistos();
  void setupResidHistos();
  void outResidHistos();

  void setUp(const edm::EventSetup& setup);

  void readEtSums();

 private:  


 
  
  // Transverse energy sum arrays
  double etsum_barl_[kBarlRings]  [kBarlWedges] [kSides];
  double etsum_endc_[kEndcWedgesX][kEndcWedgesX][kSides];
  double etsum_endc_uncorr[kEndcWedgesX][kEndcWedgesX][kSides];
  double etsumMean_barl_[kBarlRings];
  double etsumMean_endc_[kEndcEtaRings];
   
  unsigned int nhits_barl_[kBarlRings][kBarlWedges] [kSides];
  unsigned int nhits_endc_[kEndcWedgesX][kEndcWedgesX][kSides];
   
   
  double esum_barl_[kBarlRings]  [kBarlWedges] [kSides];
  double esum_endc_[kEndcWedgesX][kEndcWedgesX][kSides];

  double esumMean_barl_[kBarlRings];
  double esumMean_endc_[kEndcEtaRings];

  double k_barl_[kBarlRings];
  double k_endc_[kEndcEtaRings];
   
   // calibration const not corrected for k
  float rawconst_barl[kBarlRings][kBarlWedges][kSides];
  float rawconst_endc[kEndcWedgesX][kEndcWedgesX][kSides];   


  // calibration constants not multiplied by old ones
  float epsilon_M_barl[kBarlRings][kBarlWedges][kSides];
  float epsilon_M_endc[kEndcWedgesX][kEndcWedgesY][kSides];

  EcalGeomPhiSymHelper e_; 

  std::vector<DetId> barrelCells;
  std::vector<DetId> endcapCells;

  bool firstpass_;
  int statusThreshold_;


  bool reiteration_;
  std::string oldcalibfile_;
  
  /// the old calibration constants (when reiterating, the last ones derived)
  EcalIntercalibConstants oldCalibs_;
  
  /// calib constants that we are going to calculate
  EcalIntercalibConstants newCalibs_;
  
  
  /// initial miscalibration applied if any)
  EcalIntercalibConstants miscalib_;

  /// 
  bool have_initial_miscalib_;
  std::string initialmiscalibfile_;


  /// res miscalib histos
  std::vector<TH1F*> miscal_resid_barl_histos;
  std::vector<TH2F*> correl_barl_histos;
  
  std::vector<TH1F*> miscal_resid_endc_histos;
  std::vector<TH2F*> correl_endc_histos;

};
