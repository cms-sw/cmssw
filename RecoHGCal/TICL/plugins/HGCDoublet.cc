#include "HGCDoublet.h"

bool HGCDoublet::checkCompatibilityAndTag(std::vector<HGCDoublet> &allDoublets,
                                          const std::vector<int> &innerDoublets,
                                          const GlobalVector &refDir,
                                          float minCosTheta,
                                          float minCosPointing,
                                          bool debug) {
  int nDoublets = innerDoublets.size();
  int constexpr VSIZE = 4;
  int ok[VSIZE];
  double xi[VSIZE];
  double yi[VSIZE];
  double zi[VSIZE];
  int seedi[VSIZE];
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
      seedi[j] = otherDoublet.seedIndex();
      if (debug) {
        LogDebug("HGCDoublet") << i + j << " is doublet " << otherDoubletId << std::endl;
      }
    }
    for (int j = 0; j < vs; ++j) {
      if (seedi[j] != seedIndex_) {
        ok[j] = 0;
        continue;
      }
      ok[j] = areAligned(xi[j], yi[j], zi[j], xo, yo, zo, minCosTheta, minCosPointing, refDir, debug);
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
  for (int i = 0; i < lim; i += VSIZE)
    loop(i, VSIZE);
  loop(lim, nDoublets - lim);

  if (debug) {
    LogDebug("HGCDoublet") << "Found " << innerNeighbors_.size() << " compatible doublets out of " << nDoublets
                           << " considered" << std::endl;
  }
  return innerNeighbors_.empty();
}

int HGCDoublet::areAligned(double xi,
                           double yi,
                           double zi,
                           double xo,
                           double yo,
                           double zo,
                           float minCosTheta,
                           float minCosPointing,
                           const GlobalVector &refDir,
                           bool debug) const {
  auto dx1 = xo - xi;
  auto dy1 = yo - yi;
  auto dz1 = zo - zi;

  auto dx2 = innerX() - xi;
  auto dy2 = innerY() - yi;
  auto dz2 = innerZ() - zi;

  // inner product
  auto dot = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  auto dotsq = dot * dot;
  // magnitudes
  auto mag1sq = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  auto mag2sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;

  auto minCosTheta_sq = minCosTheta * minCosTheta;
  bool isWithinLimits = (dotsq > minCosTheta_sq * mag1sq * mag2sq);

  if (debug) {
    LogDebug("HGCDoublet") << "-- Are Aligned -- dotsq: " << dotsq << " mag1sq: " << mag1sq << " mag2sq: " << mag2sq
                           << "minCosTheta_sq:" << minCosTheta_sq << " isWithinLimits: " << isWithinLimits << std::endl;
  }

  // Now check the compatibility with the pointing origin.
  // Pointing origin is a vector obtained by the seed (track extrapolation i.e.)
  // or the direction wrt (0,0,0)
  // The compatibility is checked only for the innermost doublets: the
  // one with the outer doublets comes in by the alignment requirement of
  // the doublets themeselves

  const GlobalVector firstDoublet(dx2, dy2, dz2);
  const GlobalVector pointingDir = (seedIndex_ == -1) ? GlobalVector(innerX(), innerY(), innerZ()) : refDir;

  auto dot_pointing = pointingDir.dot(firstDoublet);
  auto dot_pointing_sq = dot_pointing * dot_pointing;
  auto mag_pointing_sq = pointingDir.mag2();
  auto minCosPointing_sq = minCosPointing * minCosPointing;
  bool isWithinLimitsPointing = (dot_pointing_sq > minCosPointing_sq * mag_pointing_sq * mag2sq);
  if (debug) {
    LogDebug("HGCDoublet") << "Pointing direction: " << pointingDir << std::endl;
    LogDebug("HGCDoublet") << "-- Are Aligned -- dot_pointing_sq: " << dot_pointing_sq
                           << " mag_pointing_sq: " << mag_pointing_sq << " mag2sq: " << mag2sq
                           << " isWithinLimitsPointing: " << isWithinLimitsPointing << std::endl;
  }
  // by squaring cosTheta and multiplying by the squares of the magnitudes
  // an equivalent comparison is made without the division and square root which are costly FP ops.
  return isWithinLimits && isWithinLimitsPointing;
}

void HGCDoublet::findNtuplets(std::vector<HGCDoublet> &allDoublets,
                              HGCntuplet &tmpNtuplet,
                              int seedIndex,
                              const bool outInDFS,
                              const unsigned int outInHops,
                              const unsigned int maxOutInHops,
                              std::vector<std::pair<unsigned int, unsigned int> > &outInToVisit) {
  if (!alreadyVisited_ && seedIndex == seedIndex_) {
    alreadyVisited_ = true;
    tmpNtuplet.push_back(theDoubletId_);
    unsigned int numberOfOuterNeighbors = outerNeighbors_.size();
    for (unsigned int i = 0; i < numberOfOuterNeighbors; ++i) {
      allDoublets[outerNeighbors_[i]].findNtuplets(
          allDoublets, tmpNtuplet, seedIndex, outInDFS, outInHops, maxOutInHops, outInToVisit);
    }
    if (outInDFS && outInHops < maxOutInHops) {
      for (auto inN : innerNeighbors_) {
        outInToVisit.emplace_back(inN, outInHops + 1);
      }
    }
  }
}
