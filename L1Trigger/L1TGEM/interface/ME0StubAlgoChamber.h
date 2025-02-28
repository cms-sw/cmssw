#ifndef L1Trigger_L1TGEM_ME0StubAlgoChamber_H
#define L1Trigger_L1TGEM_ME0StubAlgoChamber_H

#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPartition.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoMask.h"
#include <vector>
#include <cstdint>
#include <algorithm>

namespace l1t {
  namespace me0 {
    std::vector<std::vector<ME0StubPrimitive>> crossPartitionCancellation(
        std::vector<std::vector<ME0StubPrimitive>>& segments, int crossPartSegWidth);
    std::vector<ME0StubPrimitive> processChamber(const std::vector<std::vector<UInt192>>& chamberData,
                                                 const std::vector<std::vector<std::vector<int>>>& chamberBxData,
                                                 Config& config);
  }  // namespace me0
}  // namespace l1t

#endif