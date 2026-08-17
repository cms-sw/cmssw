#ifndef DataFormats_TrackingRecHitSoA_interface_OTRecHitsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_OTRecHitsSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

namespace reco {

  // Phase-2 OT stacked-module hits for stub formation, contiguous per stack (lower sensor then upper),
  // with the per-stack ranges in moduleStart.
  GENERATE_SOA_LAYOUT(OTRecHitsLayout,
                      // Local position (cm)
                      SOA_COLUMN(float, xLocal),
                      SOA_COLUMN(float, yLocal),
                      // Local position errors (cm)
                      SOA_COLUMN(float, xerrLocal),
                      SOA_COLUMN(float, yerrLocal),
                      // Global position (cm). rGlobal/iphi derived inline by consumers.
                      // The 6-elem global covariance is NOT stored: consumers recompute it by propagating
                      // the local errors through the per-sensor frame (StackedModuleGeometrySoA / CAModulesSoA).
                      // detectorIndex maps to StackedModuleGeometrySoA.
                      SOA_COLUMN(float, xGlobal),
                      SOA_COLUMN(float, yGlobal),
                      SOA_COLUMN(float, zGlobal),
                      SOA_COLUMN(uint16_t, detectorIndex),
                      // stackDetId is per-stack (OTHitModulesLayout), reachable via detectorIndex - nPixelModules.
                      // No isLower column: lower/upper follows from hit index vs upperSensorStart.
                      SOA_COLUMN(uint32_t, sensorDetId),
                      // Index into the original Phase2TrackerRecHit1DCollectionNew (truth matching).
                      SOA_COLUMN(uint32_t, origRecHitIdx),
                      // Cluster size (Phase2TrackerCluster1D::size()).
                      SOA_COLUMN(uint16_t, clusterSize))

  // Per-stack module layout: moduleStart[i] is first hit, upperSensorStart[i] where upper
  // sensor hits begin, moduleStart[i+1]-moduleStart[i] = total hits in stack i, stackDetId[i] the stack DetId.
  GENERATE_SOA_LAYOUT(OTHitModulesLayout,
                      SOA_COLUMN(uint32_t, moduleStart),
                      SOA_COLUMN(uint32_t, upperSensorStart),
                      SOA_COLUMN(uint32_t, stackDetId))

  GENERATE_SOA_BLOCKS(OTRecHitBlocksLayout,
                      SOA_BLOCK(otRecHits, OTRecHitsLayout),
                      SOA_BLOCK(otHitModules, OTHitModulesLayout))

  using OTRecHitsSoA = OTRecHitsLayout<>;
  using OTRecHitsView = OTRecHitsSoA::View;
  using OTRecHitsConstView = OTRecHitsSoA::ConstView;

  using OTHitModuleSoA = OTHitModulesLayout<>;
  using OTHitModuleView = OTHitModuleSoA::View;
  using OTHitModuleConstView = OTHitModuleSoA::ConstView;

  using OTRecHitBlocksSoA = OTRecHitBlocksLayout<>;
  using OTRecHitBlocksSoAView = OTRecHitBlocksSoA::View;
  using OTRecHitBlocksSoAConstView = OTRecHitBlocksSoA::ConstView;

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_OTRecHitsSoA_h
