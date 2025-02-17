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
    std::vector<uint64_t> mask_layer_data(const std::vector<uint64_t>& data, const Mask& mask);
    std::pair<std::vector<double>, double> calculate_centroids(const std::vector<uint64_t>& masked_data,
                                                               const std::vector<std::vector<int>>& partition_bx_data);
    int calculate_hit_count(const std::vector<uint64_t>& masked_data, bool light = false);
    int calculate_layer_count(const std::vector<uint64_t>& masked_data);
    std::vector<int> calculate_cluster_size(const std::vector<uint64_t>& data);
    std::vector<int> calculate_hits(const std::vector<uint64_t>& data);

    ME0StubPrimitive pat_unit(const std::vector<uint64_t>& data,
                              const std::vector<std::vector<int>>& bx_data,
                              int strip = 0,
                              int partition = -1,
                              std::vector<int> ly_thresh_patid = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 4, 4, 4, 4, 4},
                              std::vector<int> ly_thresh_eta = {4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4},
                              int input_max_span = 37,
                              bool skip_centroids = true,
                              int num_or = 2,
                              bool light_hit_count = true,
                              bool verbose = false);
  }  // namespace me0
}  // namespace l1t
#endif