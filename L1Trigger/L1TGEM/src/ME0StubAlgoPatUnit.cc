#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPatUnit.h"

using namespace l1t::me0;

std::vector<uint64_t> l1t::me0::mask_layer_data(const std::vector<uint64_t>& data, const Mask& mask_) {
  std::vector<uint64_t> out;
  out.reserve(static_cast<int>(data.size()));
  for (int i = 0; i < static_cast<int>(data.size()); ++i) {
    out.push_back(data[i] & mask_.mask[i]);
  }
  return out;
}

std::pair<std::vector<double>, double> l1t::me0::calculate_centroids(
    const std::vector<uint64_t>& masked_data, const std::vector<std::vector<int>>& partition_bx_data) {
  std::vector<double> centroids;
  std::vector<int> bxs;
  for (int ly = 0; ly < static_cast<int>(masked_data.size()); ++ly) {
    auto data = masked_data[ly];
    auto bx_data = partition_bx_data[ly];
    const auto temp = find_centroid(data);
    double cur_centroid = temp.first;
    std::vector<int> hits_indices = temp.second;
    centroids.push_back(cur_centroid);

    for (int hitIdx : hits_indices) {
      bxs.push_back(bx_data[hitIdx - 1]);
    }
  }
  if (static_cast<int>(bxs.size()) == 0) {
    return {centroids, -9999};
  }
  double bx_sum = std::accumulate(bxs.begin(), bxs.end(), 0.0);
  double count = bxs.size();
  return {centroids, bx_sum / count};
}

int l1t::me0::calculate_hit_count(const std::vector<uint64_t>& masked_data, bool light) {
  int tot_hit_count = 0;
  if (light) {
    for (int ly : {0, 5}) {
      int hit_ly = count_ones(masked_data[ly]);
      tot_hit_count += (hit_ly < 7) ? hit_ly : 7;
    }
  } else {
    for (uint64_t d : masked_data) {
      tot_hit_count += count_ones(d);
    }
  }
  return tot_hit_count;
}

int l1t::me0::calculate_layer_count(const std::vector<uint64_t>& masked_data) {
  int ly_count = 0;
  bool not_zero;
  for (uint64_t d : masked_data) {
    not_zero = (d != 0);
    ly_count += static_cast<int>(not_zero);
  }
  return ly_count;
}

std::vector<int> l1t::me0::calculate_cluster_size(const std::vector<uint64_t>& data) {
  std::vector<int> cluster_size_per_layer;
  cluster_size_per_layer.reserve(data.size());
  for (uint64_t x : data) {
    cluster_size_per_layer.push_back(max_cluster_size(x));
  }
  return cluster_size_per_layer;
}

std::vector<int> l1t::me0::calculate_hits(const std::vector<uint64_t>& data) {
  std::vector<int> n_hits_per_layer;
  n_hits_per_layer.reserve(data.size());
  for (uint64_t x : data) {
    n_hits_per_layer.push_back(count_ones(x));
  }
  return n_hits_per_layer;
}

ME0StubPrimitive l1t::me0::pat_unit(const std::vector<uint64_t>& data,
                                    const std::vector<std::vector<int>>& bx_data,
                                    int strip,
                                    int partition,
                                    std::vector<int> ly_thresh_patid,
                                    std::vector<int> ly_thresh_eta,
                                    int input_max_span,
                                    bool skip_centroids,
                                    int num_or,
                                    bool light_hit_count,
                                    bool verbose) {
  // construct the dynamic_patlist (we do not use default PATLIST anymore)
  // for robustness concern, other codes might use PATLIST, so we kept the default PATLIST in subfunc
  // however, this could cause inconsistent issue, becareful! OR find a way to modify PATLIST

  /*
    takes in sample data for each layer and returns best segment

    processing pipeline is

    (1) take in 6 layers of raw data
    (2) for the X (~16) patterns available, AND together the raw data with the respective pattern masks
    (3) count the # of hits in each pattern
    (4) calculate the centroids for each pattern
    (5) process segments
    (6) choose the max of all patterns
    (7) apply a layer threshold
    */

  // (2)
  // and the layer data with the respective layer mask to
  // determine how many hits are in each layer
  // this yields a map object that can be iterated over to get,
  //    for each of the N patterns, the masked []*6 layer data
  std::vector<std::vector<uint64_t>> masked_data;
  std::vector<int> pids;
  for (const Mask& M : LAYER_MASK) {
    masked_data.push_back(mask_layer_data(data, M));
    pids.push_back(M.id);
  }

  // (3) count # of hits & process centroids
  std::vector<int> hcs;
  std::vector<int> lcs;
  std::vector<std::vector<double>> centroids;
  std::vector<double> bxs;
  for (const std::vector<uint64_t>& x : masked_data) {
    hcs.push_back(calculate_hit_count(x, light_hit_count));
    lcs.push_back(calculate_layer_count(x));
    if (skip_centroids) {
      centroids.push_back({0, 0, 0, 0, 0, 0});
      bxs.push_back(-9999);
    } else {
      auto temp = calculate_centroids(x, bx_data);
      std::vector<double> cur_pattern_centroids = temp.first;
      int cur_pattern_bx = temp.second;
      centroids.push_back(cur_pattern_centroids);
      bxs.push_back(cur_pattern_bx);
    }
  }

  // (4) process segments & choose the max of all patterns
  ME0StubPrimitive best{0, 0, 0, strip, partition};
  for (int i = 0; i < static_cast<int>(hcs.size()); ++i) {
    ME0StubPrimitive seg{lcs[i], hcs[i], pids[i], strip, partition, bxs[i]};
    seg.update_quality();
    if (best.Quality() < seg.Quality()) {
      best = seg;
      best.SetCentroids(centroids[i]);
      best.update_quality();
    }
  }

  // (5) apply a layer threshold
  int ly_tresh_final;
  if (ly_thresh_patid[best.PatternId() - 1] > ly_thresh_eta[partition]) {
    ly_tresh_final = ly_thresh_patid[best.PatternId() - 1];
  } else {
    ly_tresh_final = ly_thresh_eta[partition];
  }

  if (best.LayerCount() < ly_tresh_final) {
    best.reset();
  }

  // (6) remove very wide segments
  if (best.PatternId() <= 10) {
    best.reset();
  }

  // (7) remove segments with large clusters for wide segments - ONLY NEEDED FOR PU200 - NOT USED AT THE MOEMENT
  std::vector<int> cluster_size_max_limits = {3, 6, 9, 12, 15};
  std::vector<int> n_hits_max_limits = {3, 6, 9, 12, 15};
  std::vector<int> cluster_size_counts = calculate_cluster_size(data);
  std::vector<int> n_hits_counts = calculate_hits(data);
  std::vector<int> n_layers_large_clusters = {0, 0, 0, 0, 0};
  std::vector<int> n_layers_large_hits = {0, 0, 0, 0, 0};
  for (int i = 0; i < static_cast<int>(cluster_size_counts.size()); ++i) {
    int threshold = cluster_size_max_limits[i];
    for (int l : cluster_size_counts) {
      if (l > threshold) {
        n_layers_large_clusters[i]++;
      }
    }
  }
  for (int i = 0; i < static_cast<int>(n_hits_max_limits.size()); ++i) {
    int threshold = n_hits_max_limits[i];
    for (int l : n_hits_counts) {
      if (l > threshold) {
        n_layers_large_hits[i]++;
      }
    }
  }

  best.SetMaxClusterSize(*std::max_element(cluster_size_counts.begin(), cluster_size_counts.end()));
  best.SetMaxNoise(*std::max_element(n_hits_counts.begin(), n_hits_counts.end()));

  best.SetHitCount(0);
  best.update_quality();

  return best;
}