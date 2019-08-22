// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018

#ifndef __RecoHGCal_TICL_PRbyMultiClusters_H__
#define __RecoHGCal_TICL_PRbyMultiClusters_H__
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"

#include <iostream>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

namespace ticl {
  class PatternRecognitionbyMultiClusters final : public PatternRecognitionAlgoBase {
  public:
    PatternRecognitionbyMultiClusters(const edm::ParameterSet& conf) : PatternRecognitionAlgoBase(conf) {}
    ~PatternRecognitionbyMultiClusters() override{};

    void makeTracksters(const PatternRecognitionAlgoBase::Inputs& input, std::vector<Trackster>& result) override;
  };
}  // namespace ticl
#endif
