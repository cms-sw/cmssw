#ifndef DataFormats_HGCalDigi_interface_HGCalECONTPacketInfoSoA_h
#define DataFormats_HGCalDigi_interface_HGCalECONTPacketInfoSoA_h

#include <cstdint>  // for uint8_t

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcaldigi {

  // Stage-1 per-TC energy matrix per ECON-T: 1 row (BX0 only, provided by the unpacker) x 48 columns (TC0..TC47)
  using Mtrxu16t148 = Eigen::Matrix<uint16_t, 1, 48>;
  namespace ECONTUnpackingFlags {
    constexpr uint8_t NormalUnpacking = 0, Stage1IOConversionError = 1, WrongSubpacketHeader = 2, tdaqIdxOutRange = 3;
  } // namespace ECONTUnpackingFlags

  inline constexpr bool isNotNormalECONT(uint16_t econtUnpackingFlag) {
    return !((econtUnpackingFlag >> ECONTUnpackingFlags::NormalUnpacking) & 0x1);
  }
  inline constexpr bool hasStage1IOConversionError(uint16_t econtUnpackingFlag) {
    return ((econtUnpackingFlag >> ECONTUnpackingFlags::Stage1IOConversionError) & 0x1);
  }
  inline constexpr bool hasWrongSubpacketHeader(uint16_t econtUnpackingFlag) {
    return ((econtUnpackingFlag >> ECONTUnpackingFlags::WrongSubpacketHeader) & 0x1);
  }
  inline constexpr bool hastdaqIdxOutRange(uint16_t econtUnpackingFlag) {
    return ((econtUnpackingFlag >> ECONTUnpackingFlags::tdaqIdxOutRange) & 0x1);
  }

  // generate structure of arrays (SoA) layout with Digi dataformat
  GENERATE_SOA_LAYOUT(HGCalECONTPacketInfoSoALayout,
                      // ECONT unpacking exception flag
                      // bit 0: Normal
                      // bit 1: Stage1IO conversion exception while unpacking ECON-T
                      // bit 2: Subpacket wrong header marker
                      // This will be saved to the first ECON-T in the block
                      // bit 3: TDaq index out of range
                      // This will be saved to the first ECON-T in the block
                      // bit 4: ... to be added
                      SOA_COLUMN(uint8_t, exception),
                      // Location
                      // If exception found before ECON-T, this would be 0
                      // Otherwise the 64b index of ECON-T payload start
                      SOA_COLUMN(uint32_t, location),
                      // Payload length
                      // If exception found before ECON-T, this would be 0
                      // Otherwise the payload length of the ECON-T
                      SOA_COLUMN(uint16_t, payloadLength),
                      // Stage-1 per-TC energy: 1 row (BX0) x 48 cols (TC0..TC47).
                      // Default value is -1 (0xFFFF as uint16_t) until the producer fills it;
                      // only filled cells (!= 0xFFFF) should be used downstream.
                      SOA_EIGEN_COLUMN(Mtrxu16t148, TCEnergy_Stage1),
                      //number of TCs saved
                      SOA_COLUMN(uint8_t, nTCs))

                      
  using HGCalECONTPacketInfoSoA = HGCalECONTPacketInfoSoALayout<>;
}  // namespace hgcaldigi

#endif
