#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"

using namespace l1t::me0;

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
int l1t::me0::countOnes(uint64_t x) {
  int cnt = 0;
  while (x > 0) {
    if (x & 1) {
      ++cnt;
    }
    x = (x >> 1);
  }
  return cnt;
}
int l1t::me0::maxClusterSize(uint64_t x) {
  int size = 0;
  int maxSize = 0;
  while (x > 0) {
    if ((x & 1) == 1) {
      size++;
    } else {
      if (size > maxSize) {
        maxSize = size;
      }
      size = 0;
    }
    x = x >> 1;
  }
  if (size > maxSize) {
    maxSize = size;
  }
  return maxSize;
}
UInt192 l1t::me0::setBit(int index, UInt192 num1 = UInt192(0)) {
  UInt192 num2 = (UInt192(1) << index);
  UInt192 final_v = num1 | num2;
  return final_v;
}
UInt192 l1t::me0::clearBit(int index, UInt192 num) {
  UInt192 bit = UInt192(1) & (num >> index);
  return num ^ (bit << index);
}
uint64_t l1t::me0::oneBitMask(int num) {
  uint64_t oMask = 0;
  int bitNum = 0;
  while (num != 0) {
    oMask |= (1 << bitNum);
    num = (num >> 1);
    ++bitNum;
  }
  return oMask;
}
std::vector<int> l1t::me0::findOnes(uint64_t& data) {
  std::vector<int> ones;
  int cnt = 0;
  while (data > 0) {
    if ((data & 1)) {
      ones.push_back(cnt + 1);
    }
    data >>= 1;
    ++cnt;
  }
  return ones;
}
std::pair<double, std::vector<int>> l1t::me0::findCentroid(uint64_t& data) {
  std::vector<int> ones = findOnes(data);
  if (static_cast<int>(ones.size()) == 0) {
    return {0.0, ones};
  }
  int sum = 0;
  for (int n : ones) {
    sum += n;
  }
  return {static_cast<double>(sum) / static_cast<double>(ones.size()), ones};
}
std::vector<std::vector<ME0StubPrimitive>> l1t::me0::chunk(const std::vector<ME0StubPrimitive>& inList, int n) {
  std::vector<std::vector<ME0StubPrimitive>> chunks;
  int size = inList.size();
  for (int i = 0; i < (size + n - 1) / n; ++i) {
    std::vector<ME0StubPrimitive> chunk(inList.begin() + i * n, inList.begin() + std::min((i + 1) * n, size));
    chunks.push_back(chunk);
  }
  return chunks;
}
void l1t::me0::segmentSorter(std::vector<ME0StubPrimitive>& segs, int n) {
  std::sort(segs.begin(), segs.end(), [](const ME0StubPrimitive& lhs, const ME0StubPrimitive& rhs) {
    return (lhs.quality() > rhs.quality());
  });
  segs = std::vector<ME0StubPrimitive>(segs.begin(), std::min(segs.begin() + n, segs.end()));
}
std::vector<int> l1t::me0::concatVector(const std::vector<std::vector<int>>& vec) {
  std::vector<int> cat;
  for (auto v : vec) {
    cat.insert(cat.end(), v.begin(), v.end());
  }
  return cat;
}
std::vector<ME0StubPrimitive> l1t::me0::concatVector(const std::vector<std::vector<ME0StubPrimitive>>& vec) {
  std::vector<ME0StubPrimitive> cat;
  for (auto v : vec) {
    cat.insert(cat.end(), v.begin(), v.end());
  }
  return cat;
}