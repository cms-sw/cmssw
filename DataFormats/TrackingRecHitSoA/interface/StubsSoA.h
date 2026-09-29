#ifndef DataFormats_TrackingRecHitSoA_interface_StubsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_StubsSoA_h

#include <alpaka/alpaka.hpp>

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

namespace reco {

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

  // Main stub SoA: stubs from hit pairs on stacked OT sensors (2-5mm separation).
  // Provide position, direction (dPhi/dr) and pT discrimination (bend cut).
  GENERATE_SOA_LAYOUT(StubsLayout,
                      // Azimuth of the published sensor hit (posHitIdx), same 16-bit encoding as the
                      // pixel hits: full circle mapped onto [-32768, 32767].
                      SOA_COLUMN(int16_t, iphi),

                      // Position and local errors of the published sensor hit (posHitIdx), copied
                      // from the OT rechit SoA.
                      SOA_COLUMN(float, xGlobal),
                      SOA_COLUMN(float, yGlobal),
                      SOA_COLUMN(float, zGlobal),
                      // Transverse radius of the published hit, sqrt(xGlobal^2 + yGlobal^2) in
                      // single precision.
                      SOA_COLUMN(float, rGlobal),
                      // Local position errors (variances, cm^2) of the published hit.
                      SOA_COLUMN(float, xerrLocal),
                      SOA_COLUMN(float, yerrLocal),

                      // Global CA module index of the stub's stack: pixel modules first, then the OT
                      // stacks, i.e. phase2PixelTopology::nModulesPix + (index of the stack in the
                      // StackedModuleGeometry / OT rechit module ordering), i.e. the detectorIndex
                      // of the OT rechits of the stack.
                      SOA_COLUMN(uint16_t, detectorIndex),

                      // Direction: dPhiDr = (phi_outer - phi_inner) / (r_outer - r_inner); encodes curvature for CA.
                      SOA_COLUMN(float, dPhiDr),

                      // Error on the above.
                      SOA_COLUMN(float, dPhiDrError),

                      // Same error from the precision (local-x) rows only: both azimuths are evaluated
                      // at the same along-strip coordinate, so that term is common mode in dphi and only
                      // the difference of the two projections survives. < 0 for non-stub entries.
                      SOA_COLUMN(float, dPhiDrErrorPrec),

                      // Original OT RecHit indices for track fitting.
                      SOA_COLUMN(uint32_t, lowerHitIdx),
                      SOA_COLUMN(uint32_t, upperHitIdx),

                      // The one sensor hit whose position and local errors the stub publishes: the
                      // macro-pixel hit of a PS stack, the physically inner sensor of a 2S stack.
                      SOA_COLUMN(uint32_t, posHitIdx),

                      // Packed flags: bit 0 isBarrel, bit 1 isFlat (barrel only), bit 2 isValid,
                      // bits 3-5 OT layer (0-5), bit 6 isPS, bit 7 reserved.
                      SOA_COLUMN(uint8_t, flags),

                      // Decode the packed flags of one stub.
                      SOA_CONST_ELEMENT_METHODS(
                          ALPAKA_FN_HOST_ACC bool isBarrel() const { return StubFlags::isBarrel(flags()); }

                          ALPAKA_FN_HOST_ACC bool isFlat() const { return StubFlags::isFlat(flags()); }

                          ALPAKA_FN_HOST_ACC bool isValid() const { return StubFlags::isValid(flags()); }

                          ALPAKA_FN_HOST_ACC uint8_t layer() const { return StubFlags::layer(flags()); }

                          ALPAKA_FN_HOST_ACC bool isPS() const { return StubFlags::isPS(flags()); }));

  // Stub-local index of the first stub of each outer-tracker CA module (entry i = CA module
  // phase2PixelTopology::nModulesPix + i), nOTModules + 1 entries, the last one the total number of stubs;
  // nModules() reports the block extent. moduleStart is an exclusive scan, so moduleStart[0] == 0.
  // Stubs are stored grouped by module in this order, so [moduleStart[i], moduleStart[i + 1]) is
  // contiguous.
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

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_StubsSoA_h
