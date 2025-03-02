#ifndef DataFormats_HGCalDigi_interface_HGCalFEDPacketInfoSoA_h
#define DataFormats_HGCalDigi_interface_HGCalFEDPacketInfoSoA_h

#include <cstdint>  // for uint8_t

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace hgcaldigi {

  namespace FEDUnpackingFlags {
    constexpr uint8_t NormalUnpacking = 0, GenericUnpackError = 1, ErrorSLinkHeader = 2, ErrorPayload = 3,
                      ErrorCaptureBlockHeader = 4, ActiveCaptureBlockFlags = 5, ErrorECONDHeader = 6,
                      ECONDPayloadLengthOverflow = 7, ECONDPayloadLengthMismatch = 8, ErrorSLinkTrailer = 9,
                      EarlySLinkEnd = 10;
  }  // namespace FEDUnpackingFlags

  inline constexpr bool isNotNormalFED(uint16_t fedUnpackingFlag) {
    return !((fedUnpackingFlag >> FEDUnpackingFlags::NormalUnpacking) & 0x1);
  }
  inline constexpr bool hasGenericUnpackError(uint16_t fedUnpackingFlag) {
    return ((fedUnpackingFlag >> FEDUnpackingFlags::GenericUnpackError) & 0x1);
  }
  inline constexpr bool hasHeaderUnpackError(uint16_t fedUnpackingFlag) {
    return ((fedUnpackingFlag >> FEDUnpackingFlags::ErrorSLinkHeader) & 0x1);
  }
  inline constexpr bool hasPayloadUnpackError(uint16_t fedUnpackingFlag) {
    return ((fedUnpackingFlag >> FEDUnpackingFlags::ErrorPayload) & 0x1);
  }
  inline constexpr bool hasCBHeaderError(uint16_t fedUnpackingFlag) {
    return ((fedUnpackingFlag >> FEDUnpackingFlags::ErrorCaptureBlockHeader) & 0x1);
  }
  inline constexpr bool hasCBActiveFlags(uint16_t fedUnpackingFlag) {
    return ((fedUnpackingFlag >> FEDUnpackingFlags::ActiveCaptureBlockFlags) & 0x1);
  }
  inline constexpr bool hasErrorECONDHeader(uint16_t fedUnpackingFlag) {
    return ((fedUnpackingFlag >> FEDUnpackingFlags::ErrorECONDHeader) & 0x1);
  }
  inline constexpr bool hasECONDPayloadLengthOverflow(uint16_t fedUnpackingFlag) {
    return ((fedUnpackingFlag >> FEDUnpackingFlags::ECONDPayloadLengthOverflow) & 0x1);
  }
  inline constexpr bool hasECONDPayloadLengthMismatch(uint16_t fedUnpackingFlag) {
    return ((fedUnpackingFlag >> FEDUnpackingFlags::ECONDPayloadLengthMismatch) & 0x1);
  }
  inline constexpr bool hasErrorSLinkTrailer(uint16_t fedUnpackingFlag) {
    return ((fedUnpackingFlag >> FEDUnpackingFlags::ErrorSLinkTrailer) & 0x1);
  }
  inline constexpr bool hasEarlySLinkEnd(uint16_t fedUnpackingFlag) {
    return ((fedUnpackingFlag >> FEDUnpackingFlags::EarlySLinkEnd) & 0x1);
  }

  GENERATE_SOA_LAYOUT(HGCalFEDPacketInfoSoALayout,
                      //FED unpacking flag
                      // bit 0 : unable to unpack headers
                      // bit 1 : unable to unpack data
                      // bit 2 : at least one capture block has active flags
                      SOA_COLUMN(uint16_t, FEDUnpackingFlag),
                      SOA_COLUMN(uint32_t, FEDPayload))  //number of words (char)

  using HGCalFEDPacketInfoSoA = HGCalFEDPacketInfoSoALayout<>;
}  // namespace hgcaldigi

#endif
