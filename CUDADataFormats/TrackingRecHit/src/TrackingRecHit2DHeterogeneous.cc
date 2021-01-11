#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
  cms::cuda::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template <>
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(gpuClustering::maxNumModules + 1, stream);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_hitsModuleStart, sizeof(uint32_t) * (gpuClustering::maxNumModules + 1), cudaMemcpyDefault, stream));
  return ret;
}
