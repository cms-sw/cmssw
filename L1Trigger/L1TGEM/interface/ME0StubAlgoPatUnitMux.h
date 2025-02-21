#ifndef L1Trigger_L1TGEM_ME0StubAlgoPatUnitMux_H
#define L1Trigger_L1TGEM_ME0StubAlgoPatUnitMux_H

#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPatUnit.h"
#include <vector>
#include <cstdint>
#include <cmath>

namespace l1t {
  namespace me0 {
    uint64_t parse_data(const UInt192& data, int strip, int max_span);
    std::vector<uint64_t> extract_data_window(const std::vector<UInt192>& ly_dat, int strip, int max_span);
    std::vector<int> parse_bx_data(const std::vector<int>& bx_data, int strip, int max_span);
    std::vector<std::vector<int>> extract_bx_data_window(const std::vector<std::vector<int>>& ly_dat,
                                                         int strip,
                                                         int max_span);
    std::vector<ME0StubPrimitive> pat_mux(const std::vector<UInt192>& partition_data,
                                          const std::vector<std::vector<int>>& partition_bx_data,
                                          int partition,
                                          Config& config);
  }  // namespace me0
}  // namespace l1t
#endif