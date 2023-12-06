#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/PatternMatchingLayer.h"

using namespace emtf::phase2::algo;

PatternMatchingLayer::PatternMatchingLayer(
        const EMTFContext& context
):
    context_(context)
{
    // Do Nothing
}

PatternMatchingLayer::~PatternMatchingLayer() {
    // Do Nothing
}

void PatternMatchingLayer::apply(
        const std::vector<hitmap_t>& zone_hitmaps,
        const bool& displaced_en,
        std::vector<road_collection_t>& zone_roads
) const {
    typedef ap_uint<v3::kHitmapNCols + v3::kPatternMatchingPadding * 2> padded_row_t;
    typedef ap_uint<v3::kHitmapNRows> pattern_activation_t;
    typedef std::array<pattern_activation_t, v3::kHitmapNCols> pattern_activation_collection_t;

    const padded_row_t padded_one = 1;

    auto& model = context_.model_;

    for (unsigned int i_zone = 0; i_zone < zone_hitmaps.size(); ++i_zone) { // Loop Zones
        auto& hitmap = zone_hitmaps[i_zone]; 
        auto* model_pc = &(model.zones_[i_zone].prompt_patterns);
        auto* model_ql = &(model.zones_[i_zone].prompt_quality_lut);

        if (displaced_en) {
            model_pc = &(model.zones_[i_zone].disp_patterns);
            model_ql = &(model.zones_[i_zone].disp_quality_lut);
        }

        // Initialize roads
        auto& roads = zone_roads.emplace_back();

        for (int i_col = 0; i_col < v3::kHitmapNCols; ++i_col) {
            roads[i_col].pattern = 0;
            roads[i_col].quality = 0;
        }

        // Apply patterns
        for (unsigned int i_pattern = 0; i_pattern < model_pc->size(); ++i_pattern) { // Loop Patterns
            auto& model_pat = (*model_pc)[i_pattern];

            // Initialize activations
            pattern_activation_collection_t pac;

            for (int i_col = 0; i_col < v3::kHitmapNCols; ++i_col) {
                pac[i_col] = 0;
            }

            // Build activations
            for (int i_row = 0; i_row < v3::kHitmapNRows; ++i_row) { // Loop Rows
                // Pad the row with zeros to cover cases where
                // pattern range is out of range
                auto hitmap_row = hitmap[i_row];
                auto& model_pat_row = model_pat[i_row];

                // Pad the hitmap row on both sides using kMaxPadding
                // We binary shift it to the left by kMaxPadding
                // effectively padding it to the right, and since
                // the bitwidth already includes both paddings
                // the left is also padded
                padded_row_t padded_hm_row = hitmap_row;
                padded_hm_row = padded_hm_row << v3::kPatternMatchingPadding;

                // Convert the model pattern row to a padded row
                padded_row_t padded_pat_row = 0;

                int offset = model_pat_row.begin;

                int bw = model_pat_row.end
                    - model_pat_row.begin
                    + 1; // Add 1 since it's an inclusive range

                for (int i_bit = 0; i_bit < bw; ++i_bit)
                    padded_pat_row |= (padded_one << (offset + i_bit));

                // Slide the pattern row across the hitmap and check for 'activations'
                for (int i_col = 0; i_col < v3::kHitmapNCols; ++i_col) {
                    // "AND" both rows together if the result is greater than 0
                    // there is an activation
                    padded_row_t result = padded_pat_row & padded_hm_row;

                    if (result > 0)
                        pac[i_col] = pac[i_col] | (1 << i_row);

                    // Shift the pattern row to the left, i.e. slide it across
                    padded_pat_row = padded_pat_row << 1;
                }
            } // End Loop Rows

            // Compare Activations
            // Update the road if the column's road has a smaller
            // quality than the new activation's quality.
            // Note: Since this is in a loop going from smallest pattern number
            // to the largest, cases where the quality is the same,
            // but the pattern number is larger the smaller one will be preferred
            for (int i_col = 0; i_col < v3::kHitmapNCols; ++i_col) {
                auto& activation = pac[i_col];
                auto quality = (*model_ql)[activation];

                auto& current_road = roads[i_col];

                if (current_road.quality < quality) {
                    current_road.pattern = i_pattern;
                    current_road.quality = quality;
                }
            }
        } // End Loop Patterns in Zones

        // Debug Info
        if (CONFIG.verbosity_ > 1) {
            for (int i_col = 0; i_col < v3::kHitmapNCols; ++i_col) {
                if (roads[i_col].quality == 0) {
                    continue;
                }

                std::cout
                    << "Road" 
                    << " zone " << i_zone
                    << " col " << i_col
                    << " pat " << roads[i_col].pattern
                    << " qual " << roads[i_col].quality
                    << std::endl;
            }
        }
    } // End Loop Zones
}

