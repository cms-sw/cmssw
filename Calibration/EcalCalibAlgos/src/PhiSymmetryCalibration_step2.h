#ifndef Calibration_EcalCalibAlgos_PhiSymmetryCalibration_step2_h
#define Calibration_EcalCalibAlgos_PhiSymmetryCalibration_step2_h

#include "Calibration/EcalCalibAlgos/interface/EcalGeomPhiSymHelper.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Event.h"

class TH1F;
class TH2F;

class PhiSymmetryCalibration_step2 : public edm::one::EDAnalyzer<> {
public:
  PhiSymmetryCalibration_step2(const edm::ParameterSet& iConfig);
  ~PhiSymmetryCalibration_step2() override;

  void beginJob() override;
  void endJob() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void fillHistos();
  void fillConstantsHistos();
  void setupResidHistos();
  void outResidHistos();

  void setUp(const edm::EventSetup& setup);

  void readEtSums();

private:
  const edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> channelStatusToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;

  // Transverse energy sum arrays
  double etsum_barl_[kBarlRings][kBarlWedges][kSides];
  double etsum_endc_[kEndcWedgesX][kEndcWedgesX][kSides];
  double etsum_endc_uncorr[kEndcWedgesX][kEndcWedgesX][kSides];
  double etsumMean_barl_[kBarlRings];
  double etsumMean_endc_[kEndcEtaRings];

  unsigned int nhits_barl_[kBarlRings][kBarlWedges][kSides];
  unsigned int nhits_endc_[kEndcWedgesX][kEndcWedgesX][kSides];

  double esum_barl_[kBarlRings][kBarlWedges][kSides];
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
  const int statusThreshold_;

  const bool reiteration_;
  const std::string oldcalibfile_;

  /// the old calibration constants (when reiterating, the last ones derived)
  EcalIntercalibConstants oldCalibs_;

  /// calib constants that we are going to calculate
  EcalIntercalibConstants newCalibs_;

  /// initial miscalibration applied if any)
  EcalIntercalibConstants miscalib_;

  ///
  const bool have_initial_miscalib_;
  const std::string initialmiscalibfile_;

  /// res miscalib histos
  std::vector<TH1F*> miscal_resid_barl_histos;
  std::vector<TH2F*> correl_barl_histos;

  std::vector<TH1F*> miscal_resid_endc_histos;
  std::vector<TH2F*> correl_endc_histos;
};
#endif
