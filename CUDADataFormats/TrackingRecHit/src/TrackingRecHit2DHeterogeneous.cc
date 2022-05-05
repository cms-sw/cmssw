#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

using namespace memoryPool::cuda;

template <>
memoryPool::unique_ptr<float> TrackingRecHit2DGPU::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = make_unique<float>(5 * nHits(), stream,onHost);
  cudaCheck(
      cudaMemcpyAsync(ret.get(), m_store32.get(), 5 * nHits(), cudaMemcpyDefault,stream));
  return ret;
}

template <>
memoryPool::unique_ptr<float> TrackingRecHit2DGPU::store32ToHostAsync(cudaStream_t stream) const {
  auto ret = make_unique<float>(static_cast<int>(n32) * nHits(), stream,onHost));
    cudaCheck(
      cudaMemcpyAsync(ret.get(), m_store32.get(), static_cast<int>(n32) * nHits(), cudaMemcpyDefault,stream));
  return ret;
}

template <>
memoryPool::unique_ptr<uint16_t> TrackingRecHit2DGPU::store16ToHostAsync(cudaStream_t stream) const {
  auto ret = make_unique<uint16_t>(static_cast<int>(n16) * nHits(), stream,onHost));
    cudaCheck(
      cudaMemcpyAsync(ret.get(), m_store16, static_cast<int>(n16) * nHits(), cudaMemcpyDefault,stream));
  return ret;
}

template <>
memoryPool::unique_ptr<uint32_t> TrackingRecHit2DGPU::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = make_unique<uint32_t>(nMaxModules() + 1, stream,onHost));
  cudaCheck(
      cudaMemcpyAsync(ret.get(), m_hitsModuleStart, sizeof(uint32_t) * (nMaxModules() + 1), cudaMemcpyDefault, stream));
  return ret;
}

// the only specialization needed
template <>
void TrackingRecHit2DHost::copyFromGPU(TrackingRecHit2DGPU const* input, cudaStream_t stream) {
  assert(input);
  m_store32 = input->localCoordToHostAsync(stream);
}
