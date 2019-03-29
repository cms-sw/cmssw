// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 11/2018
// Copyright CERN

#ifndef __RecoHGCal_TICL_HGCDoublet_H__
#define __RecoHGCal_TICL_HGCDoublet_H__

#include <cmath>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

class HGCDoublet {
 public:
  using HGCntuplet = std::vector<unsigned int>;

  HGCDoublet(const int innerClusterId, const int outerClusterId, const int doubletId,
             const std::vector<reco::CaloCluster> *layerClusters)
      : layerClusters_(layerClusters),
        theDoubletId_(doubletId),
        theInnerClusterId_(innerClusterId),
        theOuterClusterId_(outerClusterId),
        theInnerR_((*layerClusters)[innerClusterId].position().r()),
        theOuterR_((*layerClusters)[outerClusterId].position().r()),
        theInnerX_((*layerClusters)[innerClusterId].x()),
        theOuterX_((*layerClusters)[outerClusterId].x()),
        theInnerY_((*layerClusters)[innerClusterId].y()),
        theOuterY_((*layerClusters)[outerClusterId].y()),
        theInnerZ_((*layerClusters)[innerClusterId].z()),
        theOuterZ_((*layerClusters)[outerClusterId].z()),
        alreadyVisited_(false) {}

  double getInnerX() const { return theInnerX_; }

  double getOuterX() const { return theOuterX_; }

  double getInnerY() const { return theInnerY_; }

  double getOuterY() const { return theOuterY_; }

  double getInnerZ() const { return theInnerZ_; }

  double getOuterZ() const { return theOuterZ_; }

  double getInnerR() const { return theInnerR_; }

  double getOuterR() const { return theOuterZ_; }

  int getInnerClusterId() const { return theInnerClusterId_; }

  int getOuterClusterId() const { return theOuterClusterId_; }

  void tagAsOuterNeighbor(unsigned int otherDoublet) { theOuterNeighbors_.push_back(otherDoublet); }

  void tagAsInnerNeighbor(unsigned int otherDoublet) { theInnerNeighbors_.push_back(otherDoublet); }

  bool checkCompatibilityAndTag(std::vector<HGCDoublet> &allDoublets,
                                const std::vector<int> &innerDoublets, float minCosTheta,
                                float minCosPointing = 1., bool debug = false) {
    int nDoublets = innerDoublets.size();
    int constexpr VSIZE = 4;
    int ok[VSIZE];
    double xi[VSIZE];
    double yi[VSIZE];
    double zi[VSIZE];
    auto xo = getOuterX();
    auto yo = getOuterY();
    auto zo = getOuterZ();
    unsigned int doubletId = theDoubletId_;

    auto loop = [&](int i, int vs) {
      for (int j = 0; j < vs; ++j) {
        auto otherDoubletId = innerDoublets[i + j];
        auto &otherDoublet = allDoublets[otherDoubletId];
        xi[j] = otherDoublet.getInnerX();
        yi[j] = otherDoublet.getInnerY();
        zi[j] = otherDoublet.getInnerZ();
      }
      for (int j = 0; j < vs; ++j) {
        ok[j] = areAligned(xi[j], yi[j], zi[j], xo, yo, zo, minCosTheta, minCosPointing, debug);
        if (debug) {
          std::cout << "Are aligned for InnerDoubletId: " << i + j << " is " << ok[j] << std::endl;
        }
      }
      for (int j = 0; j < vs; ++j) {
        auto otherDoubletId = innerDoublets[i + j];
        auto &otherDoublet = allDoublets[otherDoubletId];
        if (ok[j]) {
          otherDoublet.tagAsOuterNeighbor(doubletId);
          allDoublets[doubletId].tagAsInnerNeighbor(otherDoubletId);
        }
      }
    };
    auto lim = VSIZE * (nDoublets / VSIZE);
    for (int i = 0; i < lim; i += VSIZE) loop(i, VSIZE);
    loop(lim, nDoublets - lim);

    if (debug) {
      std::cout << "Found " << theInnerNeighbors_.size() << " compatible doublets out of "
                << nDoublets << " considered" << std::endl;
    }
    return theInnerNeighbors_.empty();
  }

  int areAligned(double xi, double yi, double zi, double xo, double yo, double zo,
                 float minCosTheta, float minCosPointing, bool debug = false) {
    auto dx1 = xo - xi;
    auto dy1 = yo - yi;
    auto dz1 = zo - zi;

    auto dx2 = getInnerX() - xi;
    auto dy2 = getInnerY() - yi;
    auto dz2 = getInnerZ() - zi;

    // inner product
    auto dot = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
    // magnitudes
    auto mag1 = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
    auto mag2 = std::sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);
    // angle between the vectors
    auto cosTheta = dot / (mag1 * mag2);
    if (debug) {
      std::cout << "dot: " << dot << " mag1: " << mag1 << " mag2: " << mag2
                << " cosTheta: " << cosTheta << " isWithinLimits: " << (cosTheta > minCosTheta)
                << std::endl;
    }

    // Now check the compatibility with the pointing origin.
    // TODO(rovere): pass in also the origin, which is now fixed at (0,0,0)
    // The compatibility is checked only for the innermost doublets: the
    // one with the outer doublets comes in by the alignment requirement of
    // the doublets themeselves
    auto dot_pointing = dx2 * xi + dy2 * yi + dz2 * zi;
    auto mag_pointing = std::sqrt(xi * xi + yi * yi + zi * zi);
    auto cosTheta_pointing = dot_pointing / (mag2 * mag_pointing);
    if (debug) {
      std::cout << "dot_pointing: " << dot_pointing << " mag_pointing: " << mag_pointing
                << " mag2: " << mag2 << " cosTheta_pointing: " << cosTheta_pointing
                << " isWithinLimits: " << (cosTheta_pointing < minCosPointing) << std::endl;
    }

    return (cosTheta > minCosTheta) && (cosTheta_pointing > minCosPointing);
  }

  void findNtuplets(std::vector<HGCDoublet> &allDoublets, HGCntuplet &tmpNtuplet) {
    if (!alreadyVisited_) {
      alreadyVisited_ = true;
      tmpNtuplet.push_back(theDoubletId_);
      unsigned int numberOfOuterNeighbors = theOuterNeighbors_.size();
      for (unsigned int i = 0; i < numberOfOuterNeighbors; ++i) {
        allDoublets[theOuterNeighbors_[i]].findNtuplets(allDoublets, tmpNtuplet);
      }
    }
  }

 private:
  const std::vector<reco::CaloCluster> *layerClusters_;
  std::vector<int> theOuterNeighbors_;
  std::vector<int> theInnerNeighbors_;

  const int theDoubletId_;
  const int theInnerClusterId_;
  const int theOuterClusterId_;

  const double theInnerR_;
  const double theOuterR_;
  const double theInnerX_;
  const double theOuterX_;
  const double theInnerY_;
  const double theOuterY_;
  const double theInnerZ_;
  const double theOuterZ_;
  bool alreadyVisited_;
};

#endif /*HGCDoublet_H_ */
