#ifndef CUDADataFormats_RecHits_TrackingRecHitsDevice_h
#define CUDADataFormats_RecHits_TrackingRecHitsDevice_h

#include <cstdint>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template <typename TrackerTraits>
class TrackingRecHitSoADevice : public cms::cuda::PortableDeviceCollection<TrackingRecHitLayout<TrackerTraits>> {
public:
  using hitSoA = TrackingRecHitSoA<TrackerTraits>;
  //Need to decorate the class with the inherited portable accessors being now a template
  using cms::cuda::PortableDeviceCollection<TrackingRecHitLayout<TrackerTraits>>::view;
  using cms::cuda::PortableDeviceCollection<TrackingRecHitLayout<TrackerTraits>>::const_view;
  using cms::cuda::PortableDeviceCollection<TrackingRecHitLayout<TrackerTraits>>::buffer;
  using cms::cuda::PortableDeviceCollection<TrackingRecHitLayout<TrackerTraits>>::bufferSize;

  TrackingRecHitSoADevice() = default;  // cms::cuda::Product needs this

  using AverageGeometry = typename hitSoA::AverageGeometry;
  using ParamsOnGPU = typename hitSoA::ParamsOnGPU;

  // Constructor which specifies the SoA size
  explicit TrackingRecHitSoADevice(uint32_t nHits,
                                   int32_t offsetBPIX2,
                                   ParamsOnGPU const* cpeParams,
                                   uint32_t const* hitsModuleStart,
                                   cudaStream_t stream)
      : cms::cuda::PortableDeviceCollection<TrackingRecHitLayout<TrackerTraits>>(nHits, stream),
        offsetBPIX2_(offsetBPIX2) {
    cudaCheck(cudaMemcpyAsync(&(view().nHits()), &nHits, sizeof(uint32_t), cudaMemcpyDefault, stream));
    // hitsModuleStart is on Device
    cudaCheck(cudaMemcpyAsync(view().hitsModuleStart().data(),
                              hitsModuleStart,
                              sizeof(uint32_t) * int(TrackerTraits::numberOfModules + 1),
                              cudaMemcpyDefault,
                              stream));
    cudaCheck(cudaMemcpyAsync(&(view().offsetBPIX2()), &offsetBPIX2, sizeof(int32_t), cudaMemcpyDefault, stream));

    // cpeParams argument is a pointer to device memory, copy
    // its contents into the Layout.
    cudaCheck(cudaMemcpyAsync(&(view().cpeParams()), cpeParams, int(sizeof(ParamsOnGPU)), cudaMemcpyDefault, stream));
  }

  cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cudaStream_t stream) const {
    auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
    size_t rowSize = sizeof(float) * nHits();

    size_t srcPitch = ptrdiff_t(view().yLocal()) - ptrdiff_t(view().xLocal());
    cudaCheck(
        cudaMemcpy2DAsync(ret.get(), rowSize, view().xLocal(), srcPitch, rowSize, 4, cudaMemcpyDeviceToHost, stream));

    return ret;
  }  //move to utilities

  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const {
    auto ret = cms::cuda::make_host_unique<uint32_t[]>(TrackerTraits::numberOfModules + 1, stream);
    cudaCheck(cudaMemcpyAsync(ret.get(),
                              view().hitsModuleStart().data(),
                              sizeof(uint32_t) * (TrackerTraits::numberOfModules + 1),
                              cudaMemcpyDefault,
                              stream));
    return ret;
  }

  uint32_t nHits() const { return view().metadata().size(); }
  uint32_t offsetBPIX2() const {
    return offsetBPIX2_;
  }  //offsetBPIX2 is used on host functions so is useful to have it also stored in the class and not only in the layout
private:
  uint32_t offsetBPIX2_ = 0;
};

//Classes definition for Phase1/Phase2, to make the classes_def lighter. Not actually used in the code.
using TrackingRecHitSoADevicePhase1 = TrackingRecHitSoADevice<pixelTopology::Phase1>;
using TrackingRecHitSoADevicePhase2 = TrackingRecHitSoADevice<pixelTopology::Phase2>;
using TrackingRecHitSoADeviceHIonPhase1 = TrackingRecHitSoADevice<pixelTopology::HIonPhase1>;
using TrackingRecHitSoADeviceHIonPhase1 = TrackingRecHitSoADevice<pixelTopology::HIonPhase1>;

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
