// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 11/2018

#ifndef __RecoHGCal_TICL_HGCDoublet_H__
#define __RecoHGCal_TICL_HGCDoublet_H__

#include <cmath>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"

class HGCDoublet {
public:
  using HGCntuplet = std::vector<unsigned int>;

  HGCDoublet(const int innerClusterId,
             const int outerClusterId,
             const int doubletId,
             const std::vector<reco::CaloCluster> *layerClusters,
             const int seedIndex,
             bool areSiblingClusters = false)
      : layerClusters_(layerClusters),
        theDoubletId_(doubletId),
        innerClusterId_(innerClusterId),
        outerClusterId_(outerClusterId),
        innerR_((*layerClusters)[innerClusterId].position().r()),
        outerR_((*layerClusters)[outerClusterId].position().r()),
        innerX_((*layerClusters)[innerClusterId].x()),
        outerX_((*layerClusters)[outerClusterId].x()),
        innerY_((*layerClusters)[innerClusterId].y()),
        outerY_((*layerClusters)[outerClusterId].y()),
        innerZ_((*layerClusters)[innerClusterId].z()),
        outerZ_((*layerClusters)[outerClusterId].z()),
        seedIndex_(seedIndex),
        alreadyVisited_(false),
        areSiblingClusters_(areSiblingClusters) {}

  double innerX() const { return innerX_; }

  double outerX() const { return outerX_; }

  double innerY() const { return innerY_; }

  double outerY() const { return outerY_; }

  double innerZ() const { return innerZ_; }

  double outerZ() const { return outerZ_; }

  double innerR() const { return innerR_; }

  double outerR() const { return outerZ_; }

  int seedIndex() const { return seedIndex_; }

  int innerClusterId() const { return innerClusterId_; }

  int outerClusterId() const { return outerClusterId_; }

  bool areSiblingClusters() const { return areSiblingClusters_; }

  void tagAsOuterNeighbor(unsigned int otherDoublet) { outerNeighbors_.push_back(otherDoublet); }

  void tagAsInnerNeighbor(unsigned int otherDoublet) { innerNeighbors_.push_back(otherDoublet); }

  bool checkCompatibilityAndTag(std::vector<HGCDoublet> &allDoublets,
                                const std::vector<int> &innerDoublets,
                                const GlobalVector &refDir,
                                float minCosTheta,
                                float minCosPointing = 1.,
                                bool debug = false);

  int areAligned(double xi,
                 double yi,
                 double zi,
                 double xo,
                 double yo,
                 double zo,
                 float minCosTheta,
                 float minCosPointing,
                 const GlobalVector &refDir,
                 bool debug = false) const;

  void findNtuplets(std::vector<HGCDoublet> &allDoublets,
                    HGCntuplet &tmpNtuplet,
                    int seedIndex,
                    const bool outInDFS,
                    const unsigned int outInHops,
                    const unsigned int maxOutInHops,
                    std::vector<std::pair<unsigned int, unsigned int> > &outInToVisit);

  void setVisited(bool visited) { alreadyVisited_ = visited; }

private:
  const std::vector<reco::CaloCluster> *layerClusters_;
  std::vector<int> outerNeighbors_;
  std::vector<int> innerNeighbors_;

  const int theDoubletId_;
  const int innerClusterId_;
  const int outerClusterId_;

  const double innerR_;
  const double outerR_;
  const double innerX_;
  const double outerX_;
  const double innerY_;
  const double outerY_;
  const double innerZ_;
  const double outerZ_;
  int seedIndex_;
  bool alreadyVisited_;
  bool areSiblingClusters_;
};

#endif /*HGCDoublet_H_ */
