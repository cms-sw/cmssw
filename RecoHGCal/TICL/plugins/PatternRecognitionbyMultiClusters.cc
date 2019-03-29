#include "PatternRecognitionbyMultiClusters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void PatternRecognitionbyMultiClusters::makeTracksters(
    const edm::Event& ev, const edm::EventSetup& es,
    const std::vector<reco::CaloCluster>& layerClusters,
    const std::vector<std::pair<unsigned int, float> >& mask, std::vector<Trackster>& result) {
  LogDebug("HGCPatterRecoTrackster") << "making Tracksters" << std::endl;
}
