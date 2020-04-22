// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018

#ifndef __RecoHGCal_TICL_PRbyMultiClusters_H__
#define __RecoHGCal_TICL_PRbyMultiClusters_H__
#include "RecoHGCal/TICL/plugins/PatternRecognitionAlgoBase.h"

#include <iostream>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

namespace ticl {
  template <typename TILE>
  class PatternRecognitionbyMultiClusters final : public PatternRecognitionAlgoBaseT<TILE> {

  public:
    PatternRecognitionbyMultiClusters(const edm::ParameterSet& conf, const CacheBase* cache)
        : PatternRecognitionAlgoBaseT<TILE>(conf, cache) {}
    ~PatternRecognitionbyMultiClusters() override{};

    void makeTracksters(const typename PatternRecognitionAlgoBaseT<TILE>::Inputs& input,
                        std::vector<Trackster>& result,
                        std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation) override;
  };
}  // namespace ticl
#endif
