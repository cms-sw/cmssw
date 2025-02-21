#ifndef L1Trigger_L1TGEM_ME0StubAlgoPartition_H
#define L1Trigger_L1TGEM_ME0StubAlgoPartition_H

#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPatUnitMux.h"

namespace l1t {
  namespace me0 {
    bool is_ghost(const ME0StubPrimitive& seg,
                  const ME0StubPrimitive& comp,
                  bool check_ids = false,
                  bool check_strips = false);
    bool is_at_edge(int x, int group_width, int edge_distance);
    std::vector<ME0StubPrimitive> cancel_edges(const std::vector<ME0StubPrimitive>& segments,
                                               int group_width = 8,
                                               int ghost_width = 2,
                                               int edge_distance = 2,
                                               bool verbose = false);
    std::vector<ME0StubPrimitive> process_partition(const std::vector<UInt192>& partition_data,
                                                    const std::vector<std::vector<int>>& partition_bx_data,
                                                    int partition,
                                                    Config& config);
  }  // namespace me0
}  // namespace l1t

#endif