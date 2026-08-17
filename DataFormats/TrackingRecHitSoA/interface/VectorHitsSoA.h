#ifndef DataFormats_TrackingRecHitSoA_interface_VectorHitsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_VectorHitsSoA_h

#include <alpaka/alpaka.hpp>

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

namespace reco {

  namespace VectorHitFlags {
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
  }  // namespace VectorHitFlags

  // Main vector hit SoA: one entry per accepted pair of hits on the two sensors of a stacked OT
  // module, a few mm apart. An entry carries the position of one of the two hits, the direction of
  // the pair in the transverse plane (dPhiDr) with its error, the indices of the two hits and the
  // packed module flags. No transverse momentum is stored: a pair is accepted on its
  // parallax-corrected width in local x, a window set per layer. In a PS module with no hit on its
  // upper sensor, the lower-sensor hits are kept as position-only entries, without a bend measurement.
  GENERATE_SOA_LAYOUT(VectorHitsLayout,
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

                      // Global CA module index of the vector hit's stack: pixel modules first, then the OT
                      // stacks, i.e. phase2PixelTopology::nModulesPix + (index of the stack in the
                      // StackedModuleGeometry / OT rechit module ordering), i.e. the detectorIndex
                      // of the OT rechits of the stack.
                      SOA_COLUMN(uint16_t, detectorIndex),

                      // Direction: dPhiDr = (phi_outer - phi_inner) / dr, with dr the radial distance
                      // between the two sensors at the hit; encodes curvature for CA. 0 for an entry
                      // without a bend measurement.
                      SOA_COLUMN(float, dPhiDr),

                      // Error on the above, propagated from the three measured coordinates: the local x
                      // of the two hits and the local y of the published hit. < 0 for an entry without a
                      // bend measurement (see isVectorHit).
                      SOA_COLUMN(float, dPhiDrError),

                      // Indices in the OT rechit SoA of the pair's hits on the lower and on the upper
                      // sensor of the stack (topological order). upperHitIdx is UINT32_MAX for a
                      // position-only entry.
                      SOA_COLUMN(uint32_t, lowerHitIdx),
                      SOA_COLUMN(uint32_t, upperHitIdx),

                      // The one sensor hit whose position and local errors the vector hit publishes: the
                      // macro-pixel hit of a PS stack, the physically inner sensor of a 2S stack.
                      SOA_COLUMN(uint32_t, posHitIdx),

                      // Packed flags: bit 0 isBarrel, bit 1 isFlat (barrel only), bit 2 isValid,
                      // bits 3-5 OT layer (0-5), bit 6 isPS, bit 7 reserved.
                      SOA_COLUMN(uint8_t, flags),

                      // Decode the packed flags of one vector hit.
                      SOA_CONST_ELEMENT_METHODS(
                          ALPAKA_FN_HOST_ACC bool isBarrel() const { return VectorHitFlags::isBarrel(flags()); }

                          ALPAKA_FN_HOST_ACC bool isFlat() const { return VectorHitFlags::isFlat(flags()); }

                          ALPAKA_FN_HOST_ACC bool isValid() const { return VectorHitFlags::isValid(flags()); }

                          ALPAKA_FN_HOST_ACC uint8_t layer() const { return VectorHitFlags::layer(flags()); }

                          ALPAKA_FN_HOST_ACC bool isPS() const { return VectorHitFlags::isPS(flags()); }));

  // Vector-hit-local index of the first vector hit of each outer-tracker CA module (entry i = CA module
  // phase2PixelTopology::nModulesPix + i), nOTModules + 1 entries, the last one the total number of vector hits;
  // the collections' nModules() returns nOTModules. moduleStart is an exclusive scan, so moduleStart[0] == 0.
  // Vector hits are stored grouped by module in this order, so [moduleStart[i], moduleStart[i + 1]) is
  // contiguous.
  GENERATE_SOA_LAYOUT(VectorHitModulesLayout, SOA_COLUMN(uint32_t, moduleStart));

  GENERATE_SOA_BLOCKS(VectorHitBlocksLayout,
                      SOA_BLOCK(vectorHits, VectorHitsLayout),
                      SOA_BLOCK(vectorHitModules, VectorHitModulesLayout))

  using VectorHitsSoA = VectorHitsLayout<>;
  using VectorHitsView = VectorHitsSoA::View;
  using VectorHitsConstView = VectorHitsSoA::ConstView;

  using VectorHitModuleSoA = VectorHitModulesLayout<>;
  using VectorHitModuleView = VectorHitModuleSoA::View;
  using VectorHitModuleConstView = VectorHitModuleSoA::ConstView;

  using VectorHitBlocksSoA = VectorHitBlocksLayout<>;
  using VectorHitBlocksSoAView = VectorHitBlocksSoA::View;
  using VectorHitBlocksSoAConstView = VectorHitBlocksSoA::ConstView;

  ALPAKA_FN_HOST_ACC inline bool isVectorHit(const VectorHitsConstView &vectorHits, int32_t i) {
    return vectorHits[i].dPhiDrError() >= 0.f;
  }
  ALPAKA_FN_HOST_ACC inline bool isVectorHit(const VectorHitsConstView::const_element &vectorHit) {
    return vectorHit.dPhiDrError() >= 0.f;
  }

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_VectorHitsSoA_h
