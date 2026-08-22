#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// utility functions
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
l1t::me0::UInt192 l1t::me0::setBit(int index, l1t::me0::UInt192 num1 = l1t::me0::UInt192(0)) {
  l1t::me0::UInt192 num2 = (l1t::me0::UInt192(1) << index);
  l1t::me0::UInt192 final_v = num1 | num2;
  return final_v;
}
l1t::me0::UInt192 l1t::me0::clearBit(int index, l1t::me0::UInt192 num) {
  l1t::me0::UInt192 bit = l1t::me0::UInt192(1) & (num >> index);
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
std::vector<int> l1t::me0::findOnes(const uint64_t& data) {
  std::vector<int> ones;
  int cnt = 0;
  uint64_t temp = data;
  while (temp > 0) {
    if ((temp & 1)) {
      ones.push_back(cnt + 1);
    }
    temp >>= 1;
    ++cnt;
  }
  return ones;
}
std::pair<int, std::vector<int>> l1t::me0::findCentroid(const uint64_t& data) {
  std::vector<int> ones = l1t::me0::findOnes(data);
  if (static_cast<int>(ones.size()) == 0) {
    return {0, ones};
  }
  int sum = std::accumulate(ones.begin(), ones.end(), 0);

  double resolutionFactor =
      2.0;  // 1.0 = single strip resolution, 2.0 = half strip resolution, ...; FW is bit-retricted, so this must match the FW implementation
  double centerOfMass = resolutionFactor * static_cast<double>(sum) / static_cast<double>(ones.size());
  int roundedCenter = static_cast<int>(std::round(centerOfMass));  // FW outputs the nearest integer to the true value
  return {roundedCenter, ones};
}
std::vector<std::vector<ME0StubPrimitive>> l1t::me0::chunk(const std::vector<ME0StubPrimitive>& inList, int n) {
  std::vector<std::vector<ME0StubPrimitive>> chunks;
  int size = inList.size();
  for (int i = 0; i < (size + n - 1) / n; ++i) {
    std::vector<ME0StubPrimitive> ch(inList.begin() + i * n, inList.begin() + std::min((i + 1) * n, size));
    chunks.push_back(ch);
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
