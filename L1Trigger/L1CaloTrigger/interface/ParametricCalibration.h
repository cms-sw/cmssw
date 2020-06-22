#ifndef L1Trigger_L1CaloTrigger_ParametricCalibration_h
#define L1Trigger_L1CaloTrigger_ParametricCalibration_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <vector>
#include <cmath>
#include <iostream>

namespace l1tp2 {
  class ParametricCalibration {
  public:
    ParametricCalibration() {}
    ParametricCalibration(const edm::ParameterSet &cpset) {
      std::vector<double> etaBins = cpset.getParameter<std::vector<double>>("etaBins");
      std::vector<double> ptBins = cpset.getParameter<std::vector<double>>("ptBins");
      std::vector<double> scale = cpset.getParameter<std::vector<double>>("scale");
      etas.insert(etas.end(), etaBins.begin(), etaBins.end());
      pts.insert(pts.end(), ptBins.begin(), ptBins.end());
      scales.insert(scales.end(), scale.begin(), scale.end());
      if (cpset.existsAs<std::vector<double>>("ptMin")) {
        std::vector<double> ptMin = cpset.getParameter<std::vector<double>>("ptMin");
        ptMins.insert(ptMins.end(), ptMin.begin(), ptMin.end());
      } else {
        float ptMin = cpset.existsAs<double>("ptMin") ? cpset.getParameter<double>("ptMin") : 0;
        ptMins = std::vector<float>(etaBins.size(), ptMin);
      }
      if (cpset.existsAs<std::vector<double>>("ptMax")) {
        std::vector<double> ptMax = cpset.getParameter<std::vector<double>>("ptMax");
        ptMaxs.insert(ptMaxs.end(), ptMax.begin(), ptMax.end());
      } else {
        ptMaxs = std::vector<float>(etaBins.size(), 1e6);
      }

      if (pts.size() * etas.size() != scales.size())
        throw cms::Exception("Configuration",
                             "Bad number of calibration scales, pts.size() * etas.size() != scales.size()");
    }
    float operator()(const float pt, const float abseta) const {
      int ptBin = -1;
      for (unsigned int i = 0, n = pts.size(); i < n; ++i) {
        if (pt < pts[i]) {
          ptBin = i;
          break;
        }
      }
      int etaBin = -1;
      for (unsigned int i = 0, n = etas.size(); i < n; ++i) {
        if (abseta < etas[i]) {
          etaBin = i;
          break;
        }
      }

      if (ptBin == -1 || etaBin == -1)
        return 1;
      else
        return scales[ptBin * etas.size() + etaBin];
    }

  protected:
    std::vector<float> etas, pts, scales, ptMins, ptMaxs;
  };

};  // namespace l1tp2

#endif
