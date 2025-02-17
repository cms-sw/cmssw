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
        // map<int, vector<vector<uint64_t>>> cross_partition_cancellation(vector<vector<uint64_t>> segments);
        std::vector<std::vector<ME0StubPrimitive>> cross_partition_cancellation(std::vector<std::vector<ME0StubPrimitive>>& segments, int cross_part_seg_width);
        std::vector<ME0StubPrimitive> process_chamber(const std::vector<std::vector<UInt192>>& chamber_data,
                                                      const std::vector<std::vector<std::vector<int>>>& chamber_bx_data,
                                                      Config& config);
    }
}

#endif