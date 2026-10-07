#ifndef DataFormats_HGCalReco_interface_HGCalRecoVariations_h
#define DataFormats_HGCalReco_interface_HGCalRecoVariations_h

#include <cstdint>

#include <Eigen/Core>

namespace hgcalreco {

  // indices of *_variations columns; entries are absolute values, an undefined variation equals nominal
  namespace Variation {
    constexpr uint8_t CM_ALT = 0x0,  // DNN instead of linear common-mode correction
        SCALE_ALT = 0x1;             // not defined yet
    constexpr uint8_t Size = 0x2;
  }  // namespace Variation

  // alias keeps template commas out of the SoA macros
  using VariationVector = Eigen::Matrix<float, Variation::Size, 1>;

}  // namespace hgcalreco

#endif  // DataFormats_HGCalReco_interface_HGCalRecoVariations_h
