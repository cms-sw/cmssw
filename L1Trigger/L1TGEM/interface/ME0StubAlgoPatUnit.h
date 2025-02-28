#ifndef L1Trigger_L1TGEM_ME0StubAlgoPatUnit_H
#define L1Trigger_L1TGEM_ME0StubAlgoPatUnit_H

#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoMask.h"
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace l1t {
  namespace me0 {
    std::vector<uint64_t> maskLayerData(const std::vector<uint64_t>& data, const Mask& mask);
    std::pair<std::vector<double>, double> calculateCentroids(const std::vector<uint64_t>& maskedData,
                                                              const std::vector<std::vector<int>>& partitionBxData);
    int calculateHitCount(const std::vector<uint64_t>& maskedData, bool light = false);
    int calculateLayerCount(const std::vector<uint64_t>& maskedData);
    std::vector<int> calculateClusterSize(const std::vector<uint64_t>& data);
    std::vector<int> calculateHits(const std::vector<uint64_t>& data);

    ME0StubPrimitive patUnit(
        const std::vector<uint64_t>& data,
        const std::vector<std::vector<int>>& bxData,
        int strip = 0,
        int partition = -1,
        std::vector<int> layerThresholdPatternId = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 4, 4, 4, 4, 4},
        // layer count threshold for 17 pattern ids
        std::vector<int> layerThresholdEta = {4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4},
        // layer count threshold for 8 eta partitions + 7 "virtual" eta partitions
        int inputMaxSpan = 37,
        bool skipCentroids = true,
        int numOr = 2,
        bool lightHitCount = true,
        bool verbose = false);
  }  // namespace me0
}  // namespace l1t
#endif