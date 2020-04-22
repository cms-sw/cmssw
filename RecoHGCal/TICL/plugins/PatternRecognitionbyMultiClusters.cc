#include "PatternRecognitionbyMultiClusters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

template <typename TILE>
void ticl::PatternRecognitionbyMultiClusters<TILE>::makeTracksters(const typename PatternRecognitionAlgoBaseT<TILE>::Inputs &input,
							     std::vector<Trackster>& result,
							     std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation) {
  LogDebug("HGCPatterRecoTrackster") << "making Tracksters" << std::endl;
}
