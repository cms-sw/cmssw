#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

template <typename TrackerTraits>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DGPUT<TrackerTraits>::localCoordToHostAsync(
    cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(5 * this->nHits(), stream);
  cms::cuda::copyAsync(ret, this->m_store32, 5 * this->nHits(), stream);
  return ret;
}

template <typename TrackerTraits>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DGPUT<TrackerTraits>::store32ToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(static_cast<int>(this->n32) * this->nHits(), stream);
  cms::cuda::copyAsync(ret, this->m_store32, static_cast<int>(this->n32) * this->nHits(), stream);
  return ret;
}

template <typename TrackerTraits>
cms::cuda::host::unique_ptr<uint16_t[]> TrackingRecHit2DGPUT<TrackerTraits>::store16ToHostAsync(
    cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(static_cast<int>(this->n16) * this->nHits(), stream);
  cms::cuda::copyAsync(ret, this->m_store16, static_cast<int>(this->n16) * this->nHits(), stream);
  return ret;
}

template <typename TrackerTraits>
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DGPUT<TrackerTraits>::hitsModuleStartToHostAsync(
    cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(TrackerTraits::numberOfModules + 1, stream);
  cudaCheck(cudaMemcpyAsync(ret.get(),
                            this->m_hitsModuleStart,
                            sizeof(uint32_t) * (TrackerTraits::numberOfModules + 1),
                            cudaMemcpyDefault,
                            stream));
  return ret;
}

template class TrackingRecHit2DGPUT<pixelTopology::Phase1>;
template class TrackingRecHit2DGPUT<pixelTopology::Phase2>;

template class TrackingRecHit2DCPUT<pixelTopology::Phase1>;
template class TrackingRecHit2DCPUT<pixelTopology::Phase2>;

template class TrackingRecHit2DHostT<pixelTopology::Phase1>;
template class TrackingRecHit2DHostT<pixelTopology::Phase2>;
