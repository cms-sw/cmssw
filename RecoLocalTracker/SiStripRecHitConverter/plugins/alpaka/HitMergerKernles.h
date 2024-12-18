// #ifndef RecoLocalTracker_SiPixelRecHits_HitMergerKernels_h
// #define RecoLocalTracker_SiPixelRecHits_HitMergerKernels_h

// #include <cstdint>

// #include <alpaka/alpaka.hpp>

// #include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
// #include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
// #include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
// #include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
// #include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
// #include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
// #include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// #include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
// #include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

// namespace ALPAKA_ACCELERATOR_NAMESPACE {
//   namespace pixelgpudetails {
//     using namespace cms::alpakatools;
//     using namespace ALPAKA_ACCELERATOR_NAMESPACE::reco;
//     class HitMergerKernels {

//       using HitsView = ::reco::TrackingRecHitView;
//       using HitsConstView = ::reco::TrackingRecHitConstView;
//       using HitModulesView = ::reco::HitModuleSoAView;
//       using HitModulesConstView = ::reco::HitModuleSoAConstView;

//     public:
//       HitMergerKernels() = default;
//       ~HitMergerKernels() = default;

//       HitMergerKernels(const HitMergerKernels&) = delete;
//       HitMergerKernels(HitMergerKernels&&) = delete;
//       HitMergerKernels& operator=(const HitMergerKernels&) = delete;
//       HitMergerKernels& operator=(HitMergerKernels&&) = delete;

//       void mergeHitsAsync(SiPixelDigisSoACollection const& digis_d,
//                                                                 SiPixelClustersSoACollection const& clusters_d,
//                                                                 BeamSpotPOD const* bs_d,
//                                                                 ParamsOnDevice const* cpeParams,
//                                                                 Queue queue) const;
//     };
//   }  // namespace pixelgpudetails
// }  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// #endif  // RecoLocalTracker_SiPixelRecHits_HitMergerKernels_h
