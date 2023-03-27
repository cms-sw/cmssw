#ifndef CalibTracker_SiPixelLorentzAngle_SiPixelLorentzAngleCalibrationStruct_h
#define CalibTracker_SiPixelLorentzAngle_SiPixelLorentzAngleCalibrationStruct_h

#include "DQMServices/Core/interface/DQMStore.h"
#include <unordered_map>

namespace siPixelLACalibration {
  static constexpr float cmToum = 10000.f;

  class Chebyshev {
  public:
    Chebyshev(int n, double xmin, double xmax) : fA(xmin), fB(xmax), fT(std::vector<double>(n + 1)) {}

    double operator()(const double* xx, const double* p) {
      double x = (2.0 * xx[0] - fA - fB) / (fB - fA);
      int npar = fT.size();

      if (npar == 1)
        return p[0];
      if (npar == 2)
        return p[0] + x * p[1];
      // build the polynomials
      fT[0] = 1;
      fT[1] = x;
      for (int i = 2; i < npar; ++i) {
        fT[i] = 2 * x * fT[i - 1] - fT[i - 2];
      }
      double sum = p[0] * fT[0];
      for (int i = 1; i < npar; ++i) {
        sum += p[i] * fT[i];
      }
      return sum;
    }

  private:
    double fA;
    double fB;
    std::vector<double> fT;  // polynomial
    std::vector<double> fC;  // coefficients
  };
}  // namespace siPixelLACalibration

struct SiPixelLorentzAngleCalibrationHistograms {
public:
  SiPixelLorentzAngleCalibrationHistograms() = default;

  using MonitorMap = std::unordered_map<uint32_t, dqm::reco::MonitorElement*>;

  int nlay;
  std::vector<int> nModules_;
  std::vector<int> nLadders_;
  std::vector<std::string> BPixnewmodulename_;
  std::vector<unsigned int> BPixnewDetIds_;
  std::vector<int> BPixnewModule_;
  std::vector<int> BPixnewLayer_;

  std::vector<std::string> FPixnewmodulename_;
  std::vector<int> FPixnewDetIds_;
  std::vector<int> FPixnewDisk_;
  std::vector<int> FPixnewBlade_;
  std::unordered_map<uint32_t, std::vector<uint32_t> > detIdsList;

  MonitorMap h_drift_depth_adc_;
  MonitorMap h_drift_depth_adc2_;
  MonitorMap h_drift_depth_noadc_;
  MonitorMap h_drift_depth_;
  MonitorMap h_mean_;

  // track monitoring
  dqm::reco::MonitorElement* h_tracks_;
  dqm::reco::MonitorElement* h_trackEta_;
  dqm::reco::MonitorElement* h_trackPhi_;
  dqm::reco::MonitorElement* h_trackPt_;
  dqm::reco::MonitorElement* h_trackChi2_;

  // per-sector measurements
  dqm::reco::MonitorElement* h_bySectOccupancy_;
  dqm::reco::MonitorElement* h_bySectMeasLA_;
  dqm::reco::MonitorElement* h_bySectSetLA_;
  dqm::reco::MonitorElement* h_bySectRejectLA_;
  dqm::reco::MonitorElement* h_bySectLA_;
  dqm::reco::MonitorElement* h_bySectDeltaLA_;
  dqm::reco::MonitorElement* h_bySectChi2_;
  dqm::reco::MonitorElement* h_bySectFitStatus_;
  dqm::reco::MonitorElement* h_bySectCovMatrixStatus_;
  dqm::reco::MonitorElement* h_bySectDriftError_;

  // for fit quality
  dqm::reco::MonitorElement* h_bySectFitQuality_;

  // ouput LA maps
  std::vector<dqm::reco::MonitorElement*> h2_byLayerLA_;
  std::vector<dqm::reco::MonitorElement*> h2_byLayerDiff_;

  // FPix Minimal Cluster Size
  static constexpr int nRings_ = 2;
  static constexpr int nPanels_ = 2;
  static constexpr int nSides_ = 2;
  static constexpr int betaStartIdx_ = nRings_ * nPanels_ * nSides_;
  static constexpr int nAngles_ = 2;

  MonitorMap h_fpixAngleSize_;
  MonitorMap h_fpixMean_;
  MonitorMap h_fpixMagField_[3];

  dqm::reco::MonitorElement* h_fpixMeanHistoFitStatus_;
  dqm::reco::MonitorElement* h_fpixMinClusterSizeCotAngle_;
  dqm::reco::MonitorElement* h_fpixNhitsClusterSizeCotAngle_;
  dqm::reco::MonitorElement* h_fpixFitStatusMuH_;
  dqm::reco::MonitorElement* h_fpixMuH_;
  dqm::reco::MonitorElement* h_fpixDeltaMuH_;
  dqm::reco::MonitorElement* h_fpixRelDeltaMuH_;
};

#endif
