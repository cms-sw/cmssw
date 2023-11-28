#ifndef CUDADataFormats_RecHits_TrackingRecHitsHost_h
#define CUDADataFormats_RecHits_TrackingRecHitsHost_h

#include <cstdint>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template <typename TrackerTraits>
class TrackingRecHitSoAHost : public cms::cuda::PortableHostCollection<TrackingRecHitLayout<TrackerTraits>> {
public:
  using hitSoA = TrackingRecHitSoA<TrackerTraits>;
  //Need to decorate the class with the inherited portable accessors being now a template
  using cms::cuda::PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>::view;
  using cms::cuda::PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>::const_view;
  using cms::cuda::PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>::buffer;
  using cms::cuda::PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>::bufferSize;

  TrackingRecHitSoAHost() = default;

  using AverageGeometry = typename hitSoA::AverageGeometry;
  using ParamsOnGPU = typename hitSoA::ParamsOnGPU;
  using PhiBinnerStorageType = typename hitSoA::PhiBinnerStorageType;
  using PhiBinner = typename hitSoA::PhiBinner;

  // This SoA Host is used basically only for DQM
  // so we  just need a slim constructor
  explicit TrackingRecHitSoAHost(uint32_t nHits)
      : cms::cuda::PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>(nHits) {}

  explicit TrackingRecHitSoAHost(uint32_t nHits, cudaStream_t stream)
      : cms::cuda::PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>(nHits, stream) {}

  explicit TrackingRecHitSoAHost(uint32_t nHits,
                                 int32_t offsetBPIX2,
                                 ParamsOnGPU const* cpeParams,
                                 uint32_t const* hitsModuleStart)
      : cms::cuda::PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>(nHits), offsetBPIX2_(offsetBPIX2) {
    view().nHits() = nHits;
    std::copy(hitsModuleStart, hitsModuleStart + TrackerTraits::numberOfModules + 1, view().hitsModuleStart().begin());
    memcpy(&(view().cpeParams()), cpeParams, sizeof(ParamsOnGPU));
    view().offsetBPIX2() = offsetBPIX2;
  }

  explicit TrackingRecHitSoAHost(uint32_t nHits,
                                 int32_t offsetBPIX2,
                                 ParamsOnGPU const* cpeParams,
                                 uint32_t const* hitsModuleStart,
                                 cudaStream_t stream)
      : cms::cuda::PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>(nHits, stream),
        offsetBPIX2_(offsetBPIX2) {
    view().nHits() = nHits;
    std::copy(hitsModuleStart, hitsModuleStart + TrackerTraits::numberOfModules + 1, view().hitsModuleStart().begin());
    memcpy(&(view().cpeParams()), cpeParams, sizeof(ParamsOnGPU));
    view().offsetBPIX2() = offsetBPIX2;
  }

  uint32_t nHits() const { return view().metadata().size(); }
  uint32_t offsetBPIX2() const {
    return offsetBPIX2_;
  }  //offsetBPIX2 is used on host functions so is useful to have it also stored in the class and not only in the layout
private:
  uint32_t offsetBPIX2_ = 0;
};

using TrackingRecHitSoAHostPhase1 = TrackingRecHitSoAHost<pixelTopology::Phase1>;
using TrackingRecHitSoAHostPhase2 = TrackingRecHitSoAHost<pixelTopology::Phase2>;
using TrackingRecHitSoAHostHIonPhase1 = TrackingRecHitSoAHost<pixelTopology::HIonPhase1>;

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
