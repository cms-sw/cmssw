#ifndef CUDADataFormats_RecHits_TrackingRecHitsUtilities_h
#define CUDADataFormats_RecHits_TrackingRecHitsUtilities_h

#include <Eigen/Dense>
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "SiPixelHitStatus.h"

template <typename TrackerTraits>
struct TrackingRecHitSoA {
  using hindex_type = typename TrackerTraits::hindex_type;
  using PhiBinner = cms::cuda::HistoContainer<int16_t,
                                              256,
                                              -1,
                                              8 * sizeof(int16_t),
                                              hindex_type,
                                              TrackerTraits::numberOfLayers>;  //28 for phase2 geometry

  using PhiBinnerStorageType = typename PhiBinner::index_type;
  using AverageGeometry = pixelTopology::AverageGeometryT<TrackerTraits>;
  using ParamsOnGPU = pixelCPEforGPU::ParamsOnGPUT<TrackerTraits>;

  using HitLayerStartArray = std::array<hindex_type, TrackerTraits::numberOfLayers + 1>;
  using HitModuleStartArray = std::array<hindex_type, TrackerTraits::numberOfModules + 1>;

  //Is it better to have two split?
  GENERATE_SOA_LAYOUT(TrackingRecHitSoALayout,
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

                      SOA_SCALAR(uint32_t, nHits),
                      SOA_SCALAR(int32_t, offsetBPIX2),
                      //These above could be separated in a specific
                      //layout since they don't depends on the template
                      //for the moment I'm keeping them here
                      SOA_COLUMN(PhiBinnerStorageType, phiBinnerStorage),
                      SOA_SCALAR(HitModuleStartArray, hitsModuleStart),
                      SOA_SCALAR(HitLayerStartArray, hitsLayerStart),
                      SOA_SCALAR(ParamsOnGPU, cpeParams),
                      SOA_SCALAR(AverageGeometry, averageGeometry),
                      SOA_SCALAR(PhiBinner, phiBinner));
};

template <typename TrackerTraits>
using TrackingRecHitLayout = typename TrackingRecHitSoA<TrackerTraits>::template TrackingRecHitSoALayout<>;
template <typename TrackerTraits>
using TrackingRecHitSoAView = typename TrackingRecHitSoA<TrackerTraits>::template TrackingRecHitSoALayout<>::View;
template <typename TrackerTraits>
using TrackingRecHitSoAConstView =
    typename TrackingRecHitSoA<TrackerTraits>::template TrackingRecHitSoALayout<>::ConstView;

#endif
