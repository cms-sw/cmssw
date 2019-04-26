// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 11/2018
// Copyright CERN

#ifndef __RecoHGCal_TICL_HGCDoublet_H__
#define __RecoHGCal_TICL_HGCDoublet_H__

#include <cmath>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

class HGCDoublet {
 public:
  using HGCntuplet = std::vector<unsigned int>;

  HGCDoublet(const int innerClusterId, const int outerClusterId, const int doubletId,
             const std::vector<reco::CaloCluster> *layerClusters)
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
        alreadyVisited_(false) {}

  double innerX() const { return innerX_; }

  double outerX() const { return outerX_; }

  double innerY() const { return innerY_; }

  double outerY() const { return outerY_; }

  double innerZ() const { return innerZ_; }

  double outerZ() const { return outerZ_; }

  double innerR() const { return innerR_; }

  double outerR() const { return outerZ_; }

  int innerClusterId() const { return innerClusterId_; }

  int outerClusterId() const { return outerClusterId_; }

  void tagAsOuterNeighbor(unsigned int otherDoublet) { outerNeighbors_.push_back(otherDoublet); }

  void tagAsInnerNeighbor(unsigned int otherDoublet) { innerNeighbors_.push_back(otherDoublet); }

  bool checkCompatibilityAndTag(std::vector<HGCDoublet> &allDoublets,
                                const std::vector<int> &innerDoublets, float minCosTheta,
                                float minCosPointing = 1., bool debug = false) {
    int nDoublets = innerDoublets.size();
    int constexpr VSIZE = 4;
    int ok[VSIZE];
    double xi[VSIZE];
    double yi[VSIZE];
    double zi[VSIZE];
    auto xo = outerX();
    auto yo = outerY();
    auto zo = outerZ();
    unsigned int doubletId = theDoubletId_;

    auto loop = [&](int i, int vs) {
      for (int j = 0; j < vs; ++j) {
        auto otherDoubletId = innerDoublets[i + j];
        auto &otherDoublet = allDoublets[otherDoubletId];
        xi[j] = otherDoublet.innerX();
        yi[j] = otherDoublet.innerY();
        zi[j] = otherDoublet.innerZ();
      }
      for (int j = 0; j < vs; ++j) {
        ok[j] = areAligned(xi[j], yi[j], zi[j], xo, yo, zo, minCosTheta, minCosPointing, debug);
        if (debug) {
          LogDebug("HGCDoublet") << "Are aligned for InnerDoubletId: " << i + j << " is " << ok[j] << std::endl;
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
      LogDebug("HGCDoublet") << "Found " << innerNeighbors_.size() << " compatible doublets out of "
                << nDoublets << " considered" << std::endl;
    }
    return innerNeighbors_.empty();
  }

  int areAligned(double xi, double yi, double zi, double xo, double yo, double zo,
                 float minCosTheta, float minCosPointing, bool debug = false) const {
    auto dx1 = xo - xi;
    auto dy1 = yo - yi;
    auto dz1 = zo - zi;

    auto dx2 = innerX() - xi;
    auto dy2 = innerY() - yi;
    auto dz2 = innerZ() - zi;

    // inner product
    auto dot = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
    // magnitudes
    auto mag1 = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
    auto mag2 = std::sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);
    // angle between the vectors
    auto cosTheta = dot / (mag1 * mag2);
    if (debug) {
      LogDebug("HGCDoublet") << "dot: " << dot << " mag1: " << mag1 << " mag2: " << mag2
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
      LogDebug("HGCDoublet") << "dot_pointing: " << dot_pointing << " mag_pointing: " << mag_pointing
                << " mag2: " << mag2 << " cosTheta_pointing: " << cosTheta_pointing
                << " isWithinLimits: " << (cosTheta_pointing < minCosPointing) << std::endl;
    }

    return (cosTheta > minCosTheta) && (cosTheta_pointing > minCosPointing);
  }

  void findNtuplets(std::vector<HGCDoublet> &allDoublets, HGCntuplet &tmpNtuplet) {
    if (!alreadyVisited_) {
      alreadyVisited_ = true;
      tmpNtuplet.push_back(theDoubletId_);
      unsigned int numberOfOuterNeighbors = outerNeighbors_.size();
      for (unsigned int i = 0; i < numberOfOuterNeighbors; ++i) {
        allDoublets[outerNeighbors_[i]].findNtuplets(allDoublets, tmpNtuplet);
      }
    }
  }

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
  bool alreadyVisited_;
};

#endif /*HGCDoublet_H_ */
