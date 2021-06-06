// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 11/2018

#ifndef __RecoHGCal_TICL_HGCGraph_H__
#define __RecoHGCal_TICL_HGCGraph_H__

#include <vector>

#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"
#include "HGCDoublet.h"

template <typename TILES>
class HGCGraphT {
public:
  void makeAndConnectDoublets(const TILES &h,
                              const std::vector<TICLSeedingRegion> &regions,
                              int nEtaBins,
                              int nPhiBins,
                              const std::vector<reco::CaloCluster> &layerClusters,
                              const std::vector<float> &mask,
                              const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                              int deltaIEta,
                              int deltaIPhi,
                              float minCosThetai,
                              float maxCosPointing,
                              float root_doublet_max_distance_from_seed_squared,
                              float etaLimitIncreaseWindow,
                              int skip_layers,
                              int maxNumberOfLayers,
                              float maxDeltaTime);

  bool areTimeCompatible(int innerIdx,
                         int outerIdx,
                         const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                         float maxDeltaTime);

  std::vector<HGCDoublet> &getAllDoublets() { return allDoublets_; }
  void findNtuplets(std::vector<HGCDoublet::HGCntuplet> &foundNtuplets,
                    std::vector<int> &seedIndices,
                    const unsigned int minClustersPerNtuplet,
                    const bool outInDFS,
                    const unsigned int maxOutInHops);
  void clear() {
    allDoublets_.clear();
    theRootDoublets_.clear();
    isOuterClusterOfDoublets_.clear();
    allDoublets_.shrink_to_fit();
    theRootDoublets_.shrink_to_fit();
    isOuterClusterOfDoublets_.shrink_to_fit();
  }
  void setVerbosity(int level) { verbosity_ = level; }
  enum VerbosityLevel { None = 0, Basic, Advanced, Expert, Guru };

private:
  std::vector<HGCDoublet> allDoublets_;
  std::vector<unsigned int> theRootDoublets_;
  std::vector<std::vector<int>> isOuterClusterOfDoublets_;
  int verbosity_;
};

#endif
