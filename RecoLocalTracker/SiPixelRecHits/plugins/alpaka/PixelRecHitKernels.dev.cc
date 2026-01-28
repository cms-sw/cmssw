// C++ headers
#include <cassert>
#include <cstdint>
#include <type_traits>

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

// local headers
#include "PixelRecHitKernel.h"
#include "PixelRecHits.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::reco;

  namespace pixelgpudetails {

    template <typename TrackerTraits>
    TrackingRecHitsSoACollection PixelRecHitKernel<TrackerTraits>::makeHitsAsync(
        SiPixelDigisSoACollection const& digis_d,
        SiPixelClustersSoACollection const& clusters_d,
        BeamSpotPOD const* bs_d,
        pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* cpeParams,
        Queue queue) const {
      using namespace pixelRecHits;

      TrackingRecHitsSoACollection hits_d(queue, clusters_d);

      int activeModulesWithDigis = digis_d.nModules();

      // protect from empty events
      if (activeModulesWithDigis) {
        int threadsPerBlock = 128;
        // note: the kernel should work with an arbitrary number of blocks
        int blocks = activeModulesWithDigis;
        const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

#ifdef GPU_DEBUG
        std::cout << "launching GetHits kernel on " << alpaka::core::demangled<Acc1D> << " with " << blocks << " blocks"
                  << std::endl;
#endif
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            GetHits<TrackerTraits>{},
                            cpeParams,
                            bs_d,
                            digis_d.view(),
                            digis_d.nDigis(),
                            digis_d.nModules(),
                            clusters_d.view(),
                            hits_d.view().trackingHits());
#ifdef GPU_DEBUG
        alpaka::wait(queue);
#endif
      }

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "PixelRecHitKernel -> DONE!" << std::endl;
#endif

      return hits_d;
    }

    template class PixelRecHitKernel<pixelTopology::Phase1>;
    template class PixelRecHitKernel<pixelTopology::Phase2>;
    template class PixelRecHitKernel<pixelTopology::HIonPhase1>;

  }  // namespace pixelgpudetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
