#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

std::vector<ME0StubPrimitive> l1t::me0::PeakingManager::processSegments(const int partition,
                                                                        const std::vector<ME0StubPrimitive>& newSegs) {
  auto trig = trigger_[partition];
  auto oldSegs = segs_[0][partition];
  auto oldestSegs = segs_[1][partition];

  std::vector<ME0StubPrimitive> output;

  // peaking algorithm
  // --> trigger is only set when (oldest, old, new) = (not exist, exist, exist), and privous trigger is not set.
  // --> output is old segment when trigged, or (oldest, old, new) = (not exist, exist, not exist). Otherwise, output is empty segment.
  // ex)
  //    BX | oldest | old | new | trigger | output
  //    ---------------------------------------------------------------
  //    0  |   0    |  0  |  1  |    0    | empty
  //    1  |   0    |  1  |  0  |    0    | old (not trigged as new segment is not existed)
  //    2  |   1    |  0  |  0  |    0    | empty (not trigged as old segment is not existed)
  //    3  |   0    |  0  |  0  |    0    | empty
  // (possible case : a same segment exists for more than 2 BXs, but only trigger once when it is existed for the first time while oldest segment is not existed.)
  //    BX | oldest | old | new | trigger | output
  //    ---------------------------------------------------------------
  //    0  |   0    |  0  |  1  |    0    | empty
  //    1  |   0    |  1  |  1  |    0    | empty
  //    2  |   1    |  1  |  1  |    1    | old (trigged from BX 1)
  //    3  |   1    |  1  |  1  |    0    | empty (trigger is reset after firing at BX 2, so not trigged at BX 3)
  //    4  |   1    |  1  |  1  |    0    | empty (trigger is not set when 3 consecutive segments exist, so not trigged at BX 4)
  //    5  |   1    |  1  |  0  |    0    | empty
  //    6  |   1    |  0  |  0  |    0    | empty
  for (size_t i = 0; i < trig.size(); ++i) {
    if (trig[i]) {
      output.push_back(oldSegs[i]);
      trigger_[partition][i] = false;  // reset trigger after firing
    } else if (oldSegs[i].layerCount() > 0 && oldestSegs[i].layerCount() <= 0) {
      if (newSegs[i].layerCount() == 0) {
        output.push_back(
            oldSegs[i]);  // trigger stays false as old segment is still existed while new segment is not existed
      } else {
        output.push_back(ME0StubPrimitive(
            0, 0, 0, i, partition));  // output is empty segment as new segment is existed while old segment is existed
        trigger_[partition][i] =
            true;  // set trigger as new segment and old segment are existed while oldest segment is not existed
      }
    } else {
      output.push_back(ME0StubPrimitive(
          0, 0, 0, i, partition));  // output is empty segment as new segment is existed while old segment is existed
    }
  }

  // Update segs
  segs_[1][partition] = oldSegs;
  segs_[0][partition] = newSegs;

  return output;
}

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
  std::vector<int> ones = findOnes(data);
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
