// // C++ headers
// #include <cassert>
// #include <cstdint>
// #include <type_traits>

// // Alpaka headers
// #include <alpaka/alpaka.hpp>

// // CMSSW headers
// #include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
// #include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
// #include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
// #include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
// #include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
// #include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
// #include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

// // local headers
// #include "HitMergerKernels.h"

// //#define GPU_DEBUG

// namespace ALPAKA_ACCELERATOR_NAMESPACE {
//   using namespace cms::alpakatools;
//   using namespace ALPAKA_ACCELERATOR_NAMESPACE::reco;

//   namespace hitSoAMerger {

//     class MergeHits {
//   public:
//     template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
//     ALPAKA_FN_ACC void operator()(TAcc const& acc,
//                                   HitsView& hitsOut,
//                                   const HitsConstView& hitsIn,
//                                   HitModulesView& hitsOutModules,
//                                   const HitModulesConstView& hitsInModules) const {
 
//       ALPAKA_ASSERT_ACC(0 == mm.moduleStart()[0]);
      
//       for (int32_t i : cms::alpakatools::uniform_elements(acc, ll.metadata().size())) {
//         hitsLayerStart[i] = mm.moduleStart()[ll.layerStarts()[i]];
// #ifdef GPU_DEBUG
//         int old = i == 0 ? 0 : mm.moduleStart()[ll.layerStarts()[i - 1]];
//         printf("LayerStart %d/%d at module %d: %d - %d\n",
//                i,
//                ll.metadata().size(),
//                ll.layerStarts()[i],
//                hitsLayerStart[i],
//                hitsLayerStart[i] - old);
// #endif
//       }
//     }
//   };

//     template <typename TrackerTraits>
//     void HitMergerKernels::mergeHitsAsync(
//       const HitsConstView &hh,
//                                                                 const HitModulesConstView &mm,
//         TrackingRecHitsSoACollection& hitsOut,
//         TrackingRecHitsSoACollection const& hitsTwo,
//         Queue queue) const {
//       using namespace pixelRecHits;


//       // protect from empty events
//       if (activeModulesWithDigis) {
//         int threadsPerBlock = 128;
//         // note: the kernel should work with an arbitrary number of blocks
//         int blocks = activeModulesWithDigis;
//         const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

// #ifdef GPU_DEBUG
//         std::cout << "launching GetHits kernel on " << alpaka::core::demangled<Acc1D> << " with " << blocks << " blocks"
//                   << std::endl;
// #endif
//         alpaka::exec<Acc1D>(queue,
//                             workDiv1D,
//                             GetHits<TrackerTraits>{},
//                             cpeParams,
//                             bs_d,
//                             digis_d.view(),
//                             digis_d.nDigis(),
//                             digis_d.nModules(),
//                             clusters_d.view(),
//                             hits_d.view());
// #ifdef GPU_DEBUG
//         alpaka::wait(queue);
// #endif

//       }

// #ifdef GPU_DEBUG
//       alpaka::wait(queue);
//       std::cout << "PixelRecHitKernel -> DONE!" << std::endl;
// #endif

//       return hits_d;
//     }

//     template class PixelRecHitKernel<pixelTopology::Phase1>;
//     template class PixelRecHitKernel<pixelTopology::Phase2>;
//     template class PixelRecHitKernel<pixelTopology::HIonPhase1>;

//   }  // namespace pixelgpudetails
// }  // namespace ALPAKA_ACCELERATOR_NAMESPACE
