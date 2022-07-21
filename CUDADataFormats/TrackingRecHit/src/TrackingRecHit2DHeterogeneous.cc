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

//NB these specialization below from the base class seems to be needed since they are used by the constructor (that is in the base class)
//and if not explicilty defined the linker doesn't see them. FIXME: is there a cleaner way to do this that would still allow "partial specialization"?
//With alpaka or the unique memory pool I expect this issues to be gone

template <typename Traits, typename TrackerTraits>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DHeterogeneousT<Traits, TrackerTraits>::localCoordToHostAsync(
    cudaStream_t stream) const {
  if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
    auto ret = cms::cuda::make_host_unique<float[]>(5 * this->nHits(), stream);
    cms::cuda::copyAsync(ret, this->m_store32, 5 * this->nHits(), stream);
    return ret;
  } else {
    static_assert(true, "Intended to be used only with GPU traits.\n");
    return nullptr;
  }
}

template <typename Traits, typename TrackerTraits>
void TrackingRecHit2DHeterogeneousT<Traits, TrackerTraits>::copyFromGPU(
    TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits, TrackerTraits> const* input, cudaStream_t stream) {
  assert(input);
  if constexpr (std::is_same_v<Traits, cms::cudacompat::HostTraits>)
    this->m_store32 = input->localCoordToHostAsync(stream);
  else
    static_assert(true, "Intended to be used only with Host traits.\n");
}

template class TrackingRecHit2DGPUT<pixelTopology::Phase1>;
template class TrackingRecHit2DGPUT<pixelTopology::Phase2>;

template class TrackingRecHit2DCPUT<pixelTopology::Phase1>;
template class TrackingRecHit2DCPUT<pixelTopology::Phase2>;

template class TrackingRecHit2DHostT<pixelTopology::Phase1>;
template class TrackingRecHit2DHostT<pixelTopology::Phase2>;

//Base class instantiation
template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits, pixelTopology::Phase1>;
template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits, pixelTopology::Phase2>;

template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::CPUTraits, pixelTopology::Phase1>;
template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::CPUTraits, pixelTopology::Phase2>;

template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::HostTraits, pixelTopology::Phase1>;
template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::HostTraits, pixelTopology::Phase2>;
