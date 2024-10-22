#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsSoA_h

#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/TrackingRecHitSoA/interface/SiPixelHitStatus.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

template <typename TrackerTraits>
struct TrackingRecHitSoA {
  using hindex_type = typename TrackerTraits::hindex_type;
  using PhiBinner = cms::alpakatools::HistoContainer<int16_t,
                                                     256,
                                                     -1,  //TrackerTraits::maxNumberOfHits,
                                                     8 * sizeof(int16_t),
                                                     hindex_type,
                                                     TrackerTraits::numberOfLayers>;  //28 for phase2 geometry
  using PhiBinnerView = typename PhiBinner::View;
  using PhiBinnerStorageType = typename PhiBinner::index_type;
  using AverageGeometry = pixelTopology::AverageGeometryT<TrackerTraits>;
  using HitLayerStartArray = std::array<hindex_type, TrackerTraits::numberOfLayers + 1>;
  using HitModuleStartArray = std::array<hindex_type, TrackerTraits::numberOfModules + 1>;

  GENERATE_SOA_LAYOUT(Layout,
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
                      SOA_SCALAR(int32_t, offsetBPIX2),
                      SOA_COLUMN(PhiBinnerStorageType, phiBinnerStorage),
                      SOA_SCALAR(HitModuleStartArray, hitsModuleStart),
                      SOA_SCALAR(HitLayerStartArray, hitsLayerStart),
                      SOA_SCALAR(AverageGeometry, averageGeometry),
                      SOA_SCALAR(PhiBinner, phiBinner));
};

template <typename TrackerTraits>
using TrackingRecHitLayout = typename TrackingRecHitSoA<TrackerTraits>::template Layout<>;
template <typename TrackerTraits>
using TrackingRecHitSoAView = typename TrackingRecHitSoA<TrackerTraits>::template Layout<>::View;
template <typename TrackerTraits>
using TrackingRecHitSoAConstView = typename TrackingRecHitSoA<TrackerTraits>::template Layout<>::ConstView;

#endif
