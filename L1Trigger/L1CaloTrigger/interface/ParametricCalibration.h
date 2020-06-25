#ifndef L1Trigger_L1CaloTrigger_ParametricCalibration_h
#define L1Trigger_L1CaloTrigger_ParametricCalibration_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <vector>
#include <cmath>
#include <iostream>

namespace l1tp2 {
  class ParametricCalibration {
  public:
    ParametricCalibration() {}
    ParametricCalibration(const edm::ParameterSet& cpset);
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

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
    std::vector<float> etas, pts, scales;
  };

};  // namespace l1tp2

#endif
