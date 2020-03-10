#include "PatternRecognitionbyMultiClusters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void ticl::PatternRecognitionbyMultiClusters::makeTracksters(
    const PatternRecognitionAlgoBase::Inputs& input,
    std::vector<Trackster>& result,
    std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation) {
  LogDebug("HGCPatterRecoTrackster") << "making Tracksters" << std::endl;
}
