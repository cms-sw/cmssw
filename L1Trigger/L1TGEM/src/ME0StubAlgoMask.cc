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

Mask l1t::me0::getLayerMask(const PatternDefinition& layerPattern, int maxSpan) {
  /*
    takes in a given layer pattern and returns a list of integer bit masks
    for each layer
  */
  std::vector<std::vector<int>> mVals;
  std::vector<uint64_t> mVec;

  // for each layer, shift the provided hi and lo values for each layer from
  // pattern definition by center
  mVals.reserve(layerPattern.layers.size());
  for (HiLo layer : layerPattern.layers) {
    mVals.push_back(shiftCenter(layer, maxSpan));
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