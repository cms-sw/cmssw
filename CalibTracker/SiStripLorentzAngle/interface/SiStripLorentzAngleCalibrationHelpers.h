#ifndef CalibTracker_SiStripLorentzAngle_SiStripLorentzAngleCalibrationHelper_h
#define CalibTracker_SiStripLorentzAngle_SiStripLorentzAngleCalibrationHelper_h

// user includes
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

// system includes
#include <string>

// for ROOT
#include "TString.h"
#include "TFitResult.h"
#include "TF1.h"

namespace siStripLACalibration {

  inline std::string moduleLocationType(const uint32_t& mod, const TrackerTopology* tTopo) {
    const SiStripDetId detid(mod);
    std::string subdet = "";
    unsigned int layer = 0;
    if (detid.subDetector() == SiStripDetId::TIB) {
      subdet = "TIB";
      layer = tTopo->layer(mod);
    }

    if (detid.subDetector() == SiStripDetId::TOB) {
      subdet = "TOB";
      layer = tTopo->layer(mod);
    }

    std::string type = (detid.stereo() ? "s" : "a");
    std::string d_l_t = Form("%s_L%d%s", subdet.c_str(), layer, type.c_str());

    if (layer == 0)
      return subdet;
    return d_l_t;
  }

  // SiStripLatency::singleReadOutMode() returns
  // 1: all in peak, 0: all in deco, -1: mixed state
  enum { k_DeconvolutionMode = 0, k_PeakMode = 1 };

  //_____________________________________________________________________
  inline const std::string fieldAsString(const float& inputField) {
    std::string theMagFieldStr = std::to_string(inputField);
    size_t dotPosition = theMagFieldStr.find('.');
    if (dotPosition != std::string::npos) {
      theMagFieldStr = theMagFieldStr.substr(0, dotPosition + 2);  // +2 to include one decimal place
    }
    return theMagFieldStr;
  }

  //_____________________________________________________________________
  inline const std::string apvModeAsString(const SiStripLatency* latency) {
    if (latency) {
      switch (latency->singleReadOutMode()) {
        case k_PeakMode:
          return "PEAK";  // peak mode
        case k_DeconvolutionMode:
          return "DECO";  // deco mode
        default:
          return "UNDEF";  // undefined
      }
    } else {
      return "UNDEF";
    }
  }

  //_____________________________________________________________________
  inline double fitFunction(double* x, double* par) {
    double a = par[0];
    double thetaL = par[1];
    double b = par[2];

    double tanThetaL = std::tan(thetaL);
    double value = a * std::abs(std::tan(x[0]) - tanThetaL) + b;

    //TF1::RejectPoint();  // Reject points outside the fit range
    return value;
  }
}  // namespace siStripLACalibration
#endif
