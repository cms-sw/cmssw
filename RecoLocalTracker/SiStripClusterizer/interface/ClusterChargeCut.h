#ifndef RecoLocalTrackerSiStripClusterizerClusterChargeCut_H
#define RecoLocalTrackerSiStripClusterizerClusterChargeCut_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

inline float clusterChargeCut(const edm::ParameterSet& conf, const char* name = "clusterChargeCut") {
  return conf.getParameter<edm::ParameterSet>(name).getParameter<double>("value");
}

namespace CCC {
  // SiStripClusterChargeCutNone:    -1.0
  // SiStripClusterChargeCutTiny:   800.0
  // SiStripClusterChargeCutLoose: 1620.0
  // SiStripClusterChargeCutTight: 1945.0

  enum OP { kNone = 0, kTiny = 1, kLoose = 2, kTight = 3 };
  static constexpr std::array<float, 4> cuts = {{-1.0, 800.0, 1620.0, 1945.0}};
}  // namespace CCC

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
inline edm::ParameterSetDescription getConfigurationDescription4CCC(const CCC::OP& op) {
  edm::ParameterSetDescription desc;
  desc.add<double>("value", CCC::cuts[op]);
  return desc;
}

// this is needed to validate the configuration without explicitly setting a cut
inline edm::ParameterSetDescription getConfigurationDescription4CCCNoDefault() {
  edm::ParameterSetDescription desc;
  desc.add<double>("value");
  return desc;
}

#endif  // RecoLocalTrackerSiStripClusterizerClusterChargeCut_H
