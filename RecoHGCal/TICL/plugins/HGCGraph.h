// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 11/2018

#ifndef __RecoHGCal_TICL_HGCGraph_H__
#define __RecoHGCal_TICL_HGCGraph_H__

#include <vector>
#include "PatternRecognitionbyCAConstants.h"
#include "HGCDoublet.h"

class HGCGraph {
 public:
  void makeAndConnectDoublets(const ticl::patternbyca::Tile &h,
                              int nEtaBins, int nPhiBins,
                              const std::vector<reco::CaloCluster> &layerClusters, int deltaIEta,
                              int deltaIPhi, float minCosThetai, float maxCosPointing,
                              int missing_layers, int maxNumberOfLayers);

  std::vector<HGCDoublet> &getAllDoublets() { return allDoublets_; }
  void findNtuplets(std::vector<HGCDoublet::HGCntuplet> &foundNtuplets,
                    const unsigned int minClustersPerNtuplet);
  void clear() {
    allDoublets_.clear();
    theRootDoublets_.clear();
    theNtuplets_.clear();
    isOuterClusterOfDoublets_.clear();
  }
  void setVerbosity(int level) { verbosity_ = level; }
  enum VerbosityLevel { None = 0, Basic, Advanced, Expert, Guru };

 private:
  std::vector<HGCDoublet> allDoublets_;
  std::vector<unsigned int> theRootDoublets_;
  std::vector<std::vector<HGCDoublet *>> theNtuplets_;
  std::vector<std::vector<int>> isOuterClusterOfDoublets_;
  int verbosity_;
};

#endif
