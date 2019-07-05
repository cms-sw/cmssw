#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template<>
cudautils::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(cuda::stream_t<> &stream) const {
  edm::Service<CUDAService> cs;
  auto ret = cs->make_host_unique<float[]>(4 * nHits(), stream);
  cudautils::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template<>
cudautils::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(
    cuda::stream_t<> &stream) const {
  edm::Service<CUDAService> cs;
  auto ret = cs->make_host_unique<uint32_t[]>(2001, stream);
  cudaMemcpyAsync(ret.get(), m_hitsModuleStart, 4 * 2001, cudaMemcpyDefault, stream.id());
  return ret;
}
