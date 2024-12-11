#ifndef DataFormats_HGCalDigi_interface_HGCalECONDPacketInfoSoA_h
#define DataFormats_HGCalDigi_interface_HGCalECONDPacketInfoSoA_h

#include <cstdint>  // for uint8_t

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace hgcaldigi {

  // use Matrix for common modes
  using Matrix = Eigen ::Matrix<uint16_t, 12, 2>;
  // enum for getting ECONDFlag
  namespace ECONDFlag {
    constexpr uint8_t BITT_POS = 0, BITM_POS = 1, EBO_POS = 2, EBO_MASK = 0b11, HT_POS = 4, HT_MASK = 0b11,
                      BITE_POS = 6, BITS_POS = 7;
  }  // namespace ECONDFlag

  // functions to parse ECONDFlag
  inline bool truncatedFlag(uint8_t econdFlag) { return ((econdFlag >> hgcaldigi::ECONDFlag::BITT_POS) & 0b1); }
  inline bool matchFlag(uint8_t econdFlag) { return ((econdFlag >> hgcaldigi::ECONDFlag::BITM_POS) & 0b1); }
  inline uint8_t eboFlag(uint8_t econdFlag) {
    return ((econdFlag >> hgcaldigi::ECONDFlag::EBO_POS) & hgcaldigi::ECONDFlag::EBO_MASK);
  }
  inline uint8_t htFlag(uint8_t econdFlag) {
    return ((econdFlag >> hgcaldigi::ECONDFlag::HT_POS) & hgcaldigi::ECONDFlag::HT_MASK);
  }
  inline bool expectedFlag(uint8_t econdFlag) { return ((econdFlag >> hgcaldigi::ECONDFlag::BITE_POS) & 0b1); }
  inline bool StatFlag(uint8_t econdFlag) { return ((econdFlag >> hgcaldigi::ECONDFlag::BITS_POS) & 0b1); }

  // generate structure of arrays (SoA) layout with Digi dataformat
  GENERATE_SOA_LAYOUT(HGCalECONDPacketInfoSoALayout,
                      // Capture block information:
                      // 0b000: Normal packet
                      // 0b001: No ECOND packet. Packet was detected and discarded because too large (>250)
                      // 0b010: Packet with payload CRC error
                      // 0b011: Packet with EventID mismatch.
                      // 0b100: No ECOND packet. The event builder state machine timed-out.
                      // 0b101: No ECOND packet due to BCID and/or OrbitID mismatch.
                      // 0b110: No ECOND packet. Packet was detected but was discarded due to Main Buffer overflow.
                      SOA_COLUMN(uint8_t, cbFlag),  //cbflag
                      // ECON-D header information
                      // bit 0: Truncation flag
                      // bit 1: Match flag
                      // bit 2-3: E/B/O bits
                      // bit 4-5: H/T bits
                      // bit 6: Expected flag
                      // bit 7: logical OR of Stat for all active eRx
                      SOA_COLUMN(uint8_t, econdFlag),  //econdFlag
                      // Exception flag
                      // 0: Normal
                      // 1: Wrong S-Link header marker
                      // 2: Wrong Capture block header marker
                      // These will be saved to the first ECON-D in the block
                      // 3: Wrong ECON-D header marker
                      // 4: ECON-D payload length overflow(>469)
                      // 5: unpacked ECON-D length and payload length not match
                      // 6: S-Link trailer location error
                      // 7: S-Link End earlier
                      SOA_COLUMN(uint8_t, exception),
                      // Location
                      // If exception found before ECON-D, this would be 0
                      // Otherwise the 64b index of ECON-D header
                      SOA_COLUMN(uint32_t, location),
                      // Payload length
                      // If exception found before ECON-D, this would be 0
                      // Otherwise the payload length of the ECON-D
                      SOA_COLUMN(uint16_t, payloadLength),
                      SOA_EIGEN_COLUMN(Matrix, cm))
  using HGCalECONDPacketInfoSoA = HGCalECONDPacketInfoSoALayout<>;
}  // namespace hgcaldigi

#endif
