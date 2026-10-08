#ifndef L1Trigger_L1TGEM_ME0StubAlgoPartition_H
#define L1Trigger_L1TGEM_ME0StubAlgoPartition_H

#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPeaking.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPatUnitMux.h"

namespace l1t {
  namespace me0 {
    bool isGhost(const ME0StubPrimitive& segment,
                 const ME0StubPrimitive& comparison,
                 bool checkIds = false,
                 bool checkStrips = false);
    bool isAtEdge(int x, int groupWidth, int edgeDistance);
    std::vector<ME0StubPrimitive> cancelEdges(const std::vector<ME0StubPrimitive>& segments,
                                              int groupWidth = 8,
                                              int ghostWidth = 2,
                                              int edgeDistance = 2,
                                              bool verbose = false);
    std::vector<ME0StubPrimitive> processPartition(const std::vector<l1t::me0::UInt192>& partitionData,
                                                   const std::vector<std::vector<int>>& partitionBxData,
                                                   int partition,
                                                   l1t::me0::Config& config,
                                                   l1t::me0::PeakingManager& peakingManager);
  }  // namespace me0
}  // namespace l1t

#endif
