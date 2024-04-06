#ifndef L1Trigger_L1TMuonEndCapPhase2_OutputLayer_h
#define L1Trigger_L1TMuonEndCapPhase2_OutputLayer_h

#include <array>
#include <vector>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"

namespace emtf::phase2::algo {

  class OutputLayer {
  public:
    OutputLayer(const EMTFContext&);

    ~OutputLayer() = default;

    void apply(const int&,
               const int&,
               const int&,
               const std::map<int, int>&,
               const std::vector<track_t>&,
               const bool&,
               EMTFTrackCollection&) const;

  private:
    const EMTFContext& context_;

    std::array<float, 60> prompt_pt_calibration_lut_;
    std::array<float, 60> disp_pt_calibration_lut_;
    std::array<float, 60> disp_dxy_calibration_lut_;

    int findEMTFModeV1(const track_t::site_mask_t&) const;

    int findEMTFModeV2(const track_t::site_mask_t&) const;
  };

}  // namespace emtf::phase2::algo

#endif  // L1Trigger_L1TMuonEndCapPhase2_OutputLayer_h not defined
