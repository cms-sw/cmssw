#ifndef DataFormats_TrackingRecHitSoA_interface_StubsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_StubsSoA_h

#include <alpaka/alpaka.hpp>

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

namespace reco {

  // Main stub SoA: stubs from hit pairs on stacked OT sensors (2-5mm separation).
  // Provide position, direction (dPhi/dr) and pT discrimination (bend cut).
  GENERATE_SOA_LAYOUT(StubsLayout,
                      SOA_COLUMN(int16_t, iphi),

                      // Direction: dPhiDr = (phi_outer - phi_inner) / (r_outer - r_inner); encodes curvature for CA.
                      SOA_COLUMN(float, dPhiDr),

                      // Error on the above.
                      SOA_COLUMN(float, dPhiDrError),

                      // Same error from the precision (local-x) rows only: the along-strip cluster
                      // position is a strip centre common to both sensors, so it cancels in dphi and
                      // only the difference of the two segmentations survives. < 0 for non-stub entries.
                      SOA_COLUMN(float, dPhiDrErrorPrec),

                      // Original OT RecHit indices for track fitting.
                      SOA_COLUMN(uint32_t, lowerHitIdx),
                      SOA_COLUMN(uint32_t, upperHitIdx),

                      // Packed flags: bit 0 isBarrel, bit 1 isFlat (barrel only), bit 2 isValid,
                      // bits 3-5 OT layer (0-5), bit 6 isPS, bit 7 reserved.
                      SOA_COLUMN(uint8_t, flags));

  // Size carrier only: the extent of this block is nModules+1 and is what StubsHost/StubsDevice
  // report as nModules(); the column itself is never filled and never read.
  GENERATE_SOA_LAYOUT(StubModulesLayout, SOA_COLUMN(uint32_t, moduleStart));

  GENERATE_SOA_BLOCKS(StubBlocksLayout, SOA_BLOCK(stubs, StubsLayout), SOA_BLOCK(stubModules, StubModulesLayout))

  using StubsSoA = StubsLayout<>;
  using StubsView = StubsSoA::View;
  using StubsConstView = StubsSoA::ConstView;

  using StubModuleSoA = StubModulesLayout<>;
  using StubModuleView = StubModuleSoA::View;
  using StubModuleConstView = StubModuleSoA::ConstView;

  using StubBlocksSoA = StubBlocksLayout<>;
  using StubBlocksSoAView = StubBlocksSoA::View;
  using StubBlocksSoAConstView = StubBlocksSoA::ConstView;

  ALPAKA_FN_HOST_ACC inline bool isStub(const StubsConstView &stubs, int32_t i) {
    return stubs[i].dPhiDrError() >= 0.f;
  }
  ALPAKA_FN_HOST_ACC inline bool isStub(const StubsConstView::const_element &stub) { return stub.dPhiDrError() >= 0.f; }

  namespace StubFlags {
    constexpr uint8_t isBarrelMask = 0x01;  // Bit 0
    constexpr uint8_t isFlatMask = 0x02;    // Bit 1
    constexpr uint8_t isValidMask = 0x04;   // Bit 2
    constexpr uint8_t layerMask = 0x38;     // Bits 3-5
    constexpr uint8_t layerShift = 3;
    constexpr uint8_t isPSMask = 0x40;  // Bit 6

    inline constexpr bool isBarrel(uint8_t flags) { return (flags & isBarrelMask) != 0; }
    inline constexpr bool isFlat(uint8_t flags) { return (flags & isFlatMask) != 0; }
    inline constexpr bool isValid(uint8_t flags) { return (flags & isValidMask) != 0; }
    inline constexpr uint8_t layer(uint8_t flags) { return (flags & layerMask) >> layerShift; }
    inline constexpr bool isPS(uint8_t flags) { return (flags & isPSMask) != 0; }

    inline constexpr uint8_t makeFlags(bool barrel, bool flat, bool valid, uint8_t layerNum, bool ps) {
      return (barrel ? isBarrelMask : 0) | (flat ? isFlatMask : 0) | (valid ? isValidMask : 0) |
             ((layerNum << layerShift) & layerMask) | (ps ? isPSMask : 0);
    }
  }  // namespace StubFlags

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_StubsSoA_h
