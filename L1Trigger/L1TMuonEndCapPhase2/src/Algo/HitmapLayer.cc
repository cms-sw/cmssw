#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/HitmapLayer.h"

using namespace emtf::phase2::algo;

HitmapLayer::HitmapLayer(const EMTFContext& context) : context_(context) {}

void HitmapLayer::apply(const segment_collection_t& segments, std::vector<hitmap_t>& zone_hitmaps) const {
  const hitmap_row_t padded_one = 1;

  auto& model = context_.model_;

  // Create Images
  int n_zones = model.zones_.size();

  for (int zone_id = 0; zone_id < n_zones; ++zone_id) {  // Begin zones
    unsigned int zone_mask = (1u << zone_id);
    unsigned int tzone_mask = (1u << 0);  // Only looking at BX=0 for now

    const auto& model_hm = model.zones_[zone_id].hitmap;
    auto& hitmap = zone_hitmaps.emplace_back();
    bool hitmap_is_blank = true;

    int n_rows = model_hm.size();

    for (int row_id = 0; row_id < n_rows; ++row_id) {  // Begin loop rows

      const auto& model_hm_row = model_hm[row_id];
      auto& row = hitmap[row_id];
      row = 0;  // Clear Row Image

      for (const auto& model_hm_site : model_hm_row) {  // Begin loop sites in row

        for (const auto& model_hm_chamber : model_hm_site.chambers) {  // Begin loop chambers in site

          for (int i_ch_seg = 0; i_ch_seg < v3::kChamberSegments; ++i_ch_seg) {  // Begin loop segments

            const int seg_id = model_hm_chamber.id * v3::kChamberSegments + i_ch_seg;
            const auto& seg = segments[seg_id];

            // Short-Circuit: Must be valid
            if (seg.valid != 1) {
              continue;
            }

            // Short-Circuit: Must be same zone
            if ((seg.zones & zone_mask) != zone_mask) {
              // Debug Info
              if (this->context_.config_.verbosity_ > 4) {
                edm::LogInfo("L1TEMTFpp")
                    << "Hitmap Segment not in zone: "
                    << " zone " << zone_id << " row " << row_id << " seg_id " << seg_id << " seg_phi " << seg.phi
                    << " seg_zones " << seg.zones << " seg_tzones " << seg.tzones << std::endl;
              }

              continue;
            }

            // Short-Circuit: Must be same timezone
            if ((seg.tzones & tzone_mask) != tzone_mask) {
              // Debug Info
              if (this->context_.config_.verbosity_ > 4) {
                edm::LogInfo("L1TEMTFpp")
                    << "Hitmap Segment not in timezone: "
                    << " zone " << zone_id << " row " << row_id << " seg_id " << seg_id << " seg_phi " << seg.phi
                    << " seg_zones " << seg.zones << " seg_tzones " << seg.tzones << std::endl;
              }

              continue;
            }

            // Convert emtf_phi to col: truncate the last 4 bits, hence dividing by 16
            auto col_id = static_cast<unsigned int>(seg.phi >> v3::kHitmapColFactorLog2);

            // Debug Info
            // Seg col should be in the range specified by the model chamber
            if (this->context_.config_.verbosity_ > 4) {
              edm::LogInfo("L1TEMTFpp") << "Hitmap Segment Before Assert"
                                        << " zone " << zone_id << " row " << row_id << " col " << col_id << " seg_id "
                                        << seg_id << " seg_phi " << seg.phi << " seg_zones " << seg.zones
                                        << " seg_tzones " << seg.tzones << " ch_col_begin " << model_hm_chamber.begin
                                        << " ch_col_end " << model_hm_chamber.end << std::endl;
            }

            emtf_assert(model_hm_chamber.begin <= col_id && col_id < model_hm_chamber.end);

            // Short-Circuit: Joined chamber hitmap has more columns than the final image,
            // so we skip the columns outside of the final hitmaps's range
            // i.e. cropping the originl image
            if (!(v3::kHitmapCropColStart <= col_id && col_id < v3::kHitmapCropColStop)) {
              // Debug Info
              if (this->context_.config_.verbosity_ > 4) {
                edm::LogInfo("L1TEMTFpp") << "Hitmap Segment out of bounds: "
                                          << " zone " << zone_id << " row " << row_id << " col " << col_id << " seg_id "
                                          << seg_id << " seg_phi " << seg.phi << " seg_zones " << seg.zones
                                          << " seg_tzones " << seg.tzones << std::endl;
              }

              continue;
            }

            // Adjust col_id so kHitmapCropColStart is col 0 in the image
            col_id -= v3::kHitmapCropColStart;

            // Calculate the 0-padded int for that column and or-it into the image
            hitmap_row_t col_mask = padded_one << col_id;
            row |= col_mask;

            // Debug Info
            if (this->context_.config_.verbosity_ > 1) {
              edm::LogInfo("L1TEMTFpp") << "Hitmap Segment"
                                        << " zone " << zone_id << " row " << row_id << " col " << col_id << " seg_id "
                                        << seg_id << " seg_phi " << seg.phi << " seg_zones " << seg.zones
                                        << " seg_tzones " << seg.tzones << std::endl;
            }
          }  // End loop segments

        }  // End loop chambers in site

      }  // End loop sites in row

      // Check if hitmap is blank
      if (hitmap_is_blank && row != 0) {
        hitmap_is_blank = false;
      }
    }  // End loop rows

    // Debug Info
    if (this->context_.config_.verbosity_ > 3) {
      // Short-Circuit: the image is blank
      if (hitmap_is_blank) {
        continue;
      }

      // Pretty print
      edm::LogInfo("L1TEMTFpp") << std::endl;
      edm::LogInfo("L1TEMTFpp") << "Zone " << zone_id << " Image" << std::endl;

      for (int row_id = (model_hm.size() - 1); 0 <= row_id; --row_id) {  // Print rows in reverse order
        const auto& row = hitmap[row_id];

        edm::LogInfo("L1TEMTFpp") << row_id << " ";

        for (int col_id = 0; col_id < v3::kHitmapNCols; ++col_id) {
          hitmap_row_t pixel_mask = 1;
          pixel_mask = pixel_mask << col_id;

          bool is_present = (row & pixel_mask) == pixel_mask;

          if (is_present) {
            edm::LogInfo("L1TEMTFpp") << "X";
          } else {
            edm::LogInfo("L1TEMTFpp") << "-";
          }
        }

        edm::LogInfo("L1TEMTFpp") << std::endl;
      }

      edm::LogInfo("L1TEMTFpp") << std::endl;
    }
  }  // End loop zones
}
