#ifndef DataFormats_SiPixelDigi_interface_SiPixelDigiConstants
#define DataFormats_SiPixelDigi_interface_SiPixelDigiConstants

#include "FWCore/Utilities/interface/typedefs.h"
#include <cstdint>

using Word64 = cms_uint64_t;
using Word32 = cms_uint32_t;

namespace sipixelconstants {
  constexpr cms_uint32_t dummyDetId = 0xffffffff;

  constexpr uint32_t CRC_bits = 1;
  constexpr uint32_t DCOL_bits = 5;  // double column
  constexpr uint32_t PXID_bits = 8;  // pixel id
  constexpr uint32_t ADC_bits = 8;
  constexpr uint32_t OMIT_ERR_bits = 1;
  // GO BACK TO OLD VALUES. THE 48-CHAN FED DOES NOT NEED A NEW FORMAT 28/9/16 d.k.
  constexpr uint32_t LINK_bits = 6;  // 7;
  constexpr uint32_t ROC_bits = 5;   // 4;

  constexpr uint32_t CRC_shift = 2;
  constexpr uint32_t ADC_shift = 0;
  constexpr uint32_t PXID_shift = ADC_shift + ADC_bits;
  constexpr uint32_t DCOL_shift = PXID_shift + PXID_bits;
  constexpr uint32_t ROC_shift = DCOL_shift + DCOL_bits;
  constexpr uint32_t LINK_shift = ROC_shift + ROC_bits;
  constexpr uint32_t OMIT_ERR_shift = 20;

  constexpr uint64_t CRC_mask = ~(~Word64(0) << CRC_bits);
  constexpr uint32_t ERROR_mask = ~(~Word32(0) << ROC_bits);
  constexpr uint32_t LINK_mask = ~(~Word32(0) << LINK_bits);
  constexpr uint32_t ROC_mask = ~(~Word32(0) << ROC_bits);
  constexpr uint32_t OMIT_ERR_mask = ~(~Word32(0) << OMIT_ERR_bits);
  constexpr uint32_t DCOL_mask = ~(~Word32(0) << DCOL_bits);
  constexpr uint32_t PXID_mask = ~(~Word32(0) << PXID_bits);
  constexpr uint32_t ADC_mask = ~(~Word32(0) << ADC_bits);

  // Special for layer 1 bpix rocs 6/9/16 d.k. THIS STAYS.
  inline namespace phase1layer1 {
    constexpr uint32_t COL_bits1_l1 = 6;
    constexpr uint32_t ROW_bits1_l1 = 7;
    constexpr uint32_t ROW_shift = ADC_shift + ADC_bits;
    constexpr uint32_t COL_shift = ROW_shift + ROW_bits1_l1;
    constexpr uint32_t COL_mask = ~(~Word32(0) << COL_bits1_l1);
    constexpr uint32_t ROW_mask = ~(~Word32(0) << ROW_bits1_l1);
  }  // namespace phase1layer1

  // constexpr functions are available in device code (GPU) as well
  inline namespace functions {
    inline constexpr uint32_t getLink(uint32_t ww) { return ((ww >> LINK_shift) & LINK_mask); }
    inline constexpr uint32_t getROC(uint32_t ww) { return ((ww >> ROC_shift) & ROC_mask); }
    inline constexpr uint32_t getADC(uint32_t ww) { return ((ww >> ADC_shift) & ADC_mask); }
    inline constexpr uint32_t getCol(uint32_t ww) { return ((ww >> COL_shift) & COL_mask); }
    inline constexpr uint32_t getRow(uint32_t ww) { return ((ww >> ROW_shift) & ROW_mask); }
    inline constexpr uint32_t getDCol(uint32_t ww) { return ((ww >> DCOL_shift) & DCOL_mask); }
    inline constexpr uint32_t getPxId(uint32_t ww) { return ((ww >> PXID_shift) & PXID_mask); }
  }  // namespace functions
}  // namespace sipixelconstants

#endif  // DataFormats_SiPixelDigi_interface_SiPixelDigiConstants
