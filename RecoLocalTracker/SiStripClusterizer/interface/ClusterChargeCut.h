#ifndef RecoLocalTrackerSiStripClusterizerClusterChargeCut_H
#define RecoLocalTrackerSiStripClusterizerClusterChargeCut_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

inline float clusterChargeCut(const edm::ParameterSet& conf, const char* name = "clusterChargeCut") {
  return conf.getParameter<edm::ParameterSet>(name).getParameter<double>("value");
}

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
inline void fillPSetDescription4CCC(edm::ParameterSetDescription& iDesc) {
  // HLTSiStripClusterChargeCutNone:    -1.0
  // HLTSiStripClusterChargeCutTiny:   800.0
  // HLTSiStripClusterChargeCutLoose: 1620.0
  // HLTSiStripClusterChargeCutTight: 1945.0
  iDesc.add<double>("value", 1620.0);
}

inline edm::ParameterSetDescription getFilledConfigurationDescription4CCC() {
  edm::ParameterSetDescription desc;
  fillPSetDescription4CCC(desc);
  return desc;
}

inline edm::ParameterSetDescription getFilledConfigurationDescription4CCCNoDefault() {
  edm::ParameterSetDescription desc;
  desc.add<double>("value");
  return desc;
}

#endif  // RecoLocalTrackerSiStripClusterizerClusterChargeCut_H
