#include "L1Trigger/L1TGEM/interface/ME0StubAlgoMask.h"

using namespace l1t::me0;

std::vector<int> l1t::me0::shiftCenter(const HiLo& layer, int maxSpan) {
  /*
    Patterns are defined as a +hi and -lo around a center point of a pattern.

    e.g. for a pattern 37 strips wide, there is a central strip,
    and 18 strips to the left and right of it.

    This patterns shifts from a +hi and -lo around the central strip, to an offset +hi and -lo.

    e.g. for (hi, lo) = (1, -1) and a window of 37, this will return (17,19)
  */
  int center = std::floor(maxSpan / 2);
  int hi = layer.hi + center;
  int lo = layer.lo + center;
  std::vector<int> out = {lo, hi};
  return out;
}

uint64_t l1t::me0::setHighBits(const std::vector<int>& loHiPair) {
  /*
    Given a high bit and low bit, this function will return a bitmask with all the bits in
    between the high and low set to 1
  */
  int lo = loHiPair[0], hi = loHiPair[1];
  uint64_t out = std::pow(2, (hi - lo + 1)) - 1;
  out <<= lo;
  return out;
}

Mask l1t::me0::getLayerMask(const PatternDefinition& layerPattern, const std::vector<int> layerSpans) {
  /*
    takes in a given layer pattern and returns a list of integer bit masks
    for each layer
  */
  std::vector<std::vector<int>> mVals;
  std::vector<uint64_t> mVec;

  // for each layer, shift the provided hi and lo values for each layer from
  // pattern definition by center
  mVals.reserve(layerPattern.layers.size());
  for (int idxLy = 0; idxLy < static_cast<int>(layerPattern.layers.size()); ++idxLy) {
    mVals.push_back(shiftCenter(layerPattern.layers[idxLy], layerSpans[idxLy]));
  }

  // use the high and low indices to determine where the high bits must go for
  // each layer
  mVec.reserve(mVals.size());
  for (const std::vector<int>& x : mVals) {
    mVec.push_back(setHighBits(x));
  }

  Mask mask_{layerPattern.id, mVec};
  return mask_;
}
//define functions to generate patterns
HiLo l1t::me0::mirrorHiLo(const HiLo& layer) {
  HiLo mirrored{-1 * (layer.lo), -1 * (layer.hi)};
  return mirrored;
}
PatternDefinition l1t::me0::mirrorPatternDefinition(const PatternDefinition& pattern, int id) {
  std::vector<HiLo> layers_;
  layers_.reserve(pattern.layers.size());
  for (HiLo l : pattern.layers) {
    layers_.push_back(mirrorHiLo(l));
  }
  PatternDefinition mirrored{id, layers_};
  return mirrored;
}
std::vector<HiLo> l1t::me0::createPatternLayer(double lower, double upper) {
  std::vector<HiLo> layerList;
  double hi, lo;
  int hi_i, lo_i;
  for (int i = 0; i < 6; ++i) {
    if (i < 3) {
      hi = lower * (i - 2.5);
      lo = upper * (i - 2.5);
    } else {
      hi = upper * (i - 2.5);
      lo = lower * (i - 2.5);
    }
    if (std::abs(hi) < 0.1) {
      hi = 0.0f;
    }
    if (std::abs(lo) < 0.1) {
      lo = 0.0f;
    }
    hi_i = std::ceil(hi);
    lo_i = std::floor(lo);
    layerList.push_back(HiLo{hi_i, lo_i});
  }
  return layerList;
}
std::vector<int> l1t::me0::calculateLayerSpans(const std::vector<PatternDefinition>& patternList) {
  std::vector<int> layerSpans;
  layerSpans.reserve(6);
  for (int idxLy = 0; idxLy < 6; ++idxLy) {
    int high = 0;
    for (const PatternDefinition& pattern : patternList) {
      high = std::max(high, pattern.layers[idxLy].hi);
    }
    layerSpans.push_back(2 * high +
                         1);  // span = 2*hi + 1, since hi is defined as the number of strips above the center strip
  }
  return layerSpans;
}
std::vector<int> l1t::me0::calculatePatternSpans(const std::vector<PatternDefinition>& patternList) {
  std::vector<int> patternSpans;
  patternSpans.reserve(patternList.size());
  for (const PatternDefinition& pattern : patternList) {
    int high = std::max(pattern.layers[0].hi, pattern.layers[5].hi);
    int low = std::min(pattern.layers[0].lo, pattern.layers[5].lo);
    patternSpans.push_back(high - low + 1);
  }
  return patternSpans;
}
std::vector<std::vector<int>> l1t::me0::calculatePatternOffsets(const std::vector<PatternDefinition>& patternList,
                                                                const std::vector<int>& patternSpans,
                                                                const std::vector<int>& layerSpans) {
  std::vector<std::vector<int>> patternOffsets;
  for (int idxPat = 0; idxPat < static_cast<int>(patternList.size()); ++idxPat) {
    int maxSpan = layerSpans[0] / 2;
    std::vector<int> shiftLeft;
    shiftLeft.reserve(6);
    for (int idxLy = 0; idxLy < 6; ++idxLy) {
      shiftLeft.push_back(maxSpan - layerSpans[idxLy] / 2);
    }
    int shiftRight = maxSpan - patternSpans[idxPat] / 2;
    std::vector<int> offsets;
    offsets.reserve(6);
    for (int idxLy = 0; idxLy < 6; ++idxLy) {
      offsets.push_back(shiftLeft[idxLy] - shiftRight);
    }
    patternOffsets.push_back(offsets);
  }
  return patternOffsets;
}