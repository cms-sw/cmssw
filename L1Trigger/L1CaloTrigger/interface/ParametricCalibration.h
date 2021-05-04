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

    float operator()(const float pt, const float abseta) const;

  protected:
    std::vector<float> etas, pts, scales;
  };

};  // namespace l1tp2

#endif
