#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DataUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/RoadSortingLayer.h"

using namespace emtf::phase2::algo;

RoadSortingLayer::RoadSortingLayer(const EMTFContext& context) : context_(context) {}

void RoadSortingLayer::apply(const unsigned int& first_n,
                             const std::vector<road_collection_t>& zone_roads,
                             std::vector<road_t>& best_roads) const {
  // Find the best roads from each zone
  std::vector<road_t> top_roads;

  for (unsigned int i_zone = 0; i_zone < zone_roads.size(); ++i_zone) {  // Loop Zones

    auto& roads = zone_roads[i_zone];

    // Suppress qualities of non local maximum
    road_collection_t suppressed_roads;

    {
      const int last_col = v3::kHitmapNCols - 1;

      for (unsigned int i_col = 0; i_col < v3::kHitmapNCols; ++i_col) {
        bool is_local_max = true;
        bool is_last_col = i_col == last_col;
        bool is_first_col = i_col == 0;

        // If this is not the last column, compare it with the next column's road
        // If this column has better or equal quality than the next, this is still the local max
        if (is_local_max && !is_last_col) {
          is_local_max &= (roads[i_col].quality >= roads[i_col + 1].quality);
        }

        // If this is not the first column, compare it with the previous column's road
        // If this column has better quality than the previous, this is still the local max
        if (is_local_max && !is_first_col) {
          is_local_max &= (roads[i_col].quality > roads[i_col - 1].quality);
        }

        // Suppress qualities
        if (is_local_max) {
          suppressed_roads[i_col].zone = i_zone;
          suppressed_roads[i_col].col = i_col;
          suppressed_roads[i_col].pattern = roads[i_col].pattern;
          suppressed_roads[i_col].quality = roads[i_col].quality;
        } else {
          // Debug Info
          if (this->context_.config_.verbosity_ > 2 && roads[i_col].quality > 0) {
            edm::LogInfo("L1TEMTFpp") << "Road Suppressed"
                                      << " zone " << i_zone << " col " << i_col << " pat " << roads[i_col].pattern
                                      << " qual " << roads[i_col].quality << std::endl;
          }

          // Suppress
          suppressed_roads[i_col].zone = i_zone;
          suppressed_roads[i_col].col = i_col;
          suppressed_roads[i_col].pattern = roads[i_col].pattern;
          suppressed_roads[i_col].quality = 0;
        }
      }
    }

    // Keep best of every pair
    const int keep_n_roads = v3::kHitmapNCols / 2;

    road_t roads_kept[keep_n_roads];

    {
      for (unsigned int i_col = 0; i_col < keep_n_roads; ++i_col) {
        bool is_single = (i_col * 2 + 1) >= v3::kHitmapNCols;

        if (is_single || suppressed_roads[i_col * 2].quality > 0) {
          roads_kept[i_col] = suppressed_roads[i_col * 2];
        } else {
          roads_kept[i_col] = suppressed_roads[i_col * 2 + 1];
        }

        if (this->context_.config_.verbosity_ > 2 && roads_kept[i_col].quality > 0) {
          edm::LogInfo("L1TEMTFpp") << "Road Kept"
                                    << " zone " << roads_kept[i_col].zone << " col " << roads_kept[i_col].col << " pat "
                                    << roads_kept[i_col].pattern << " qual " << roads_kept[i_col].quality << std::endl;
        }
      }
    }

    // Mergesort-reduce to n best roads
    // This will sort descending order (higher-value means lower-index) and keep the first n roads

    // Sort the first 32 cols since there are 144 columns and we wish to sort powers of 2, therefore 128 to keep priorities.
    data::mergesort(
        roads_kept, 32, 16, [](const road_t& lhs, const road_t& rhs) -> int { return lhs.quality < rhs.quality; });

    // Shift everything 16 cols to the left
    for (unsigned int i = 16; i < keep_n_roads; ++i) {
      roads_kept[i] = roads_kept[i + 16];
    }

    // Merge-sort the remaining 128 cols
    data::mergesort(roads_kept, 128, first_n, [](const road_t& lhs, const road_t& rhs) -> int {
      return lhs.quality < rhs.quality;
    });

    // Collect best roads
    for (unsigned int i_col = 0; i_col < first_n; ++i_col) {
      top_roads.push_back(roads_kept[i_col]);
    }
  }  // End Loop Zones

  // Debug Info
  if (this->context_.config_.verbosity_ > 2) {
    for (const auto& road : top_roads) {
      // Short-Circuit: Skip quality-0 roads
      if (road.quality == 0) {
        continue;
      }

      edm::LogInfo("L1TEMTFpp") << "Top Road"
                                << " zone " << road.zone << " col " << road.col << " pat " << road.pattern << " qual "
                                << road.quality << std::endl;
    }
  }

  // Mergesort-reduce to n best roads
  // This will sort descending order (higher-value means lower-index) and keep the first n roads

  // Sort the first 8 cols since there are 12 cols and we wish to sort powers of 2, therefore 8 to keep priorities
  data::mergesort(
      &top_roads[0], 8, 4, [](const road_t& lhs, const road_t& rhs) -> int { return lhs.quality < rhs.quality; });

  // Shift everything 4 cols to the left
  for (unsigned int i = 4; i < top_roads.size(); ++i) {
    top_roads[i] = top_roads[i + 4];
  }

  // Merge-sort remaining 8 cols
  data::mergesort(
      &top_roads[0], 8, first_n, [](const road_t& lhs, const road_t& rhs) -> int { return lhs.quality < rhs.quality; });

  // Collect best roads
  for (unsigned int i_road = 0; i_road < first_n; ++i_road) {
    const auto& road = top_roads[i_road];

    best_roads.push_back(road);

    // Debug Info
    if (this->context_.config_.verbosity_ > 1 && road.quality > 0) {
      edm::LogInfo("L1TEMTFpp") << "Best Road " << i_road << " zone " << road.zone << " col " << road.col << " pat "
                                << road.pattern << " qual " << road.quality << std::endl;
    }
  }
}
