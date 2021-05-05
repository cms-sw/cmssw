#ifndef DataFormats_SiPixelDigi_interface_SiPixelDigiConstants
#define DataFormats_SiPixelDigi_interface_SiPixelDigiConstants

#include <stdint.h>
#include "FWCore/Utilities/interface/typedefs.h"

using Word64 = cms_uint64_t;
using Word32 = cms_uint32_t;

namespace sipixelconstants {
  constexpr uint32_t CRC_bits = 1;
  constexpr uint32_t LINK_bits = 6;
  constexpr uint32_t ROC_bits = 5;
  constexpr uint32_t DCOL_bits = 5;
  constexpr uint32_t PXID_bits = 8;
  constexpr uint32_t ADC_bits = 8;
  constexpr uint32_t OMIT_ERR_bits = 1;

  constexpr uint32_t CRC_shift = 2;
  constexpr uint32_t ADC_shift = 0;
  constexpr uint32_t PXID_shift = ADC_shift + ADC_bits;
  constexpr uint32_t DCOL_shift = PXID_shift + PXID_bits;
  constexpr uint32_t ROC_shift = DCOL_shift + DCOL_bits;
  constexpr uint32_t LINK_shift = ROC_shift + ROC_bits;
  constexpr uint32_t OMIT_ERR_shift = 20;

  constexpr cms_uint32_t dummyDetId = 0xffffffff;

  constexpr Word64 CRC_mask = ~(~Word64(0) << CRC_bits);
  constexpr Word32 ERROR_mask = ~(~Word32(0) << ROC_bits);
  constexpr Word32 LINK_mask = ~(~Word32(0) << LINK_bits);
  constexpr Word32 ROC_mask = ~(~Word32(0) << ROC_bits);
  constexpr Word32 OMIT_ERR_mask = ~(~Word32(0) << OMIT_ERR_bits);
}  // namespace sipixelconstants

#endif  // DataFormats_SiPixelDigi_interface_SiPixelDigiConstants
