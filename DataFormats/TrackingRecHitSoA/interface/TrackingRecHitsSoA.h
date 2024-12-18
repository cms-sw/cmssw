#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsSoA_h

#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/TrackingRecHitSoA/interface/SiPixelHitStatus.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

namespace reco {
  
  GENERATE_SOA_LAYOUT(TrackingHitsLayout,
                      SOA_COLUMN(float, xLocal),
                      SOA_COLUMN(float, yLocal),
                      SOA_COLUMN(float, xerrLocal),
                      SOA_COLUMN(float, yerrLocal),
                      SOA_COLUMN(float, xGlobal),
                      SOA_COLUMN(float, yGlobal),
                      SOA_COLUMN(float, zGlobal),
                      SOA_COLUMN(float, rGlobal),
                      SOA_COLUMN(int16_t, iphi),
                      SOA_COLUMN(SiPixelHitStatusAndCharge, chargeAndStatus),
                      SOA_COLUMN(int16_t, clusterSizeX),
                      SOA_COLUMN(int16_t, clusterSizeY),
                      SOA_COLUMN(uint16_t, detectorIndex),
                      SOA_SCALAR(int32_t, offsetBPIX2));

  GENERATE_SOA_LAYOUT(HitModulesLayout,
               SOA_COLUMN(uint32_t, moduleStart)
  );
               
  #ifdef CA_TRIPLETS_HOLE
  GENERATE_SOA_LAYOUT(AverageGeometryLayout,
    SOA_COLUMN(float, ladderZ),
    SOA_COLUMN(float, ladderX),
    SOA_COLUMN(float, ladderY),
    SOA_COLUMN(float, ladderR),
    SOA_COLUMN(float, ladderMinZ),
    SOA_COLUMN(float, ladderMaxZ),
    SOA_SCALAR(int32_t, endCapZPos),
    SOA_SCALAR(int32_t, endCapZNeg)
  )
  #endif

  using TrackingRecHitSoA = TrackingHitsLayout<>;
  using TrackingRecHitView = TrackingRecHitSoA::View;
  using TrackingRecHitConstView = TrackingRecHitSoA::ConstView;

  using HitModuleSoA = HitModulesLayout<>;
  using HitModuleSoAView = HitModuleSoA::View;
  using HitModuleSoAConstView = HitModuleSoA::ConstView;

#ifdef CA_TRIPLETS_HOLE
  using AverageGeometrySoA = AverageGeometryLayout<>;
  using AverageGeometryView = AverageGeometrySoA::View;
  using AverageGeometryConstView = AverageGeometrySoA::ConstView;
#endif

};

#endif
