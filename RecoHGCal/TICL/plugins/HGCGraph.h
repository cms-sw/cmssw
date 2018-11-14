// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 11/2018
// Copyright CERN

#ifndef __RecoHGCal_TICL_HGCGraph_H__
#define __RecoHGCal_TICL_HGCGraph_H__

#include <vector>
#include "HGCDoublet.h"

class HGCGraph
{
public:
  void makeAndConnectDoublets(const std::vector<std::vector<std::vector<unsigned int>>> &h, int nEtaBins,
                              int nPhiBins, const std::vector<reco::CaloCluster> &layerClusters, int deltaIEta, int deltaIPhi, float minCosTheta);

  std::vector<HGCDoublet> &getAllDoublets() { return allDoublets_; }
  void findNtuplets(std::vector<HGCDoublet::HGCntuplet> &foundNtuplets, const unsigned int minClustersPerNtuplet);
  void clear(){
    allDoublets_.clear();
    theRootDoublets_.clear();
    theNtuplets_.clear();
    isOuterClusterOfDoublets_.clear();
  }

private:
  std::vector<HGCDoublet> allDoublets_;
  std::vector<unsigned int> theRootDoublets_;
  std::vector<std::vector<HGCDoublet *>> theNtuplets_;
  std::vector<std::vector<int>> isOuterClusterOfDoublets_;
};

#endif
