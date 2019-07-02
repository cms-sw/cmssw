#include "PatternRecognitionbyMultiClusters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void ticl::PatternRecognitionbyMultiClusters::makeTracksters(const edm::Event& ev,
                                                             const edm::EventSetup& es,
                                                             const std::vector<reco::CaloCluster>& layerClusters,
                                                             const std::vector<float>& mask,
                                                             const edm::ValueMap<float>& layerClustersTime,
                                                             const TICLLayerTiles& tiles,
                                                             std::vector<Trackster>& result) {
  LogDebug("HGCPatterRecoTrackster") << "making Tracksters" << std::endl;
}
