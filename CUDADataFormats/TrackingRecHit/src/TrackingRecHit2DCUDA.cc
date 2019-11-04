#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

template <>
cudautils::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cudautils::make_host_unique<float[]>(4 * nHits(), stream);
  cudautils::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template <>
cudautils::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = cudautils::make_host_unique<uint32_t[]>(2001, stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), m_hitsModuleStart, 4 * 2001, cudaMemcpyDefault, stream));
  return ret;
}
