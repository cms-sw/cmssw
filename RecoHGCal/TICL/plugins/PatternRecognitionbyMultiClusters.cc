#include "PatternRecognitionbyMultiClusters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

template <typename TILES>
void ticl::PatternRecognitionbyMultiClusters<TILES>::makeTracksters(
    const typename PatternRecognitionAlgoBaseT<TILES>::Inputs& input,
    std::vector<Trackster>& result,
    std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation) {
  LogDebug("HGCPatterRecoTrackster") << "making Tracksters" << std::endl;
}
