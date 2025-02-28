#ifndef L1Trigger_L1TGEM_ME0StubAlgoPatUnitMux_H
#define L1Trigger_L1TGEM_ME0StubAlgoPatUnitMux_H

#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPatUnit.h"
#include <vector>
#include <cstdint>
#include <cmath>

namespace l1t {
  namespace me0 {
    uint64_t parseData(const UInt192& data, int strip, int maxSpan);
    std::vector<uint64_t> extractDataWindow(const std::vector<UInt192>& layerData, int strip, int maxSpan);
    std::vector<int> parseBxData(const std::vector<int>& bxData, int strip, int maxSpan);
    std::vector<std::vector<int>> extractBxDataWindow(const std::vector<std::vector<int>>& layerData,
                                                      int strip,
                                                      int maxSpan);
    std::vector<ME0StubPrimitive> patMux(const std::vector<UInt192>& partitionData,
                                         const std::vector<std::vector<int>>& partitionBxData,
                                         int partition,
                                         Config& config);
  }  // namespace me0
}  // namespace l1t
#endif