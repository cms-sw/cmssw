#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t feds, size_t nelements, cuda::stream_t<>& stream) {
  edm::Service<CUDAService> cs;

  moduleStart_d     = cs->make_device_unique<uint32_t[]>(nelements+1, stream);
  clus_d            = cs->make_device_unique< int32_t[]>(feds, stream);
  clusInModule_d    = cs->make_device_unique<uint32_t[]>(nelements, stream);
  moduleId_d        = cs->make_device_unique<uint32_t[]>(nelements, stream);
  clusModuleStart_d = cs->make_device_unique<uint32_t[]>(nelements+1, stream);

  auto view = cs->make_host_unique<DeviceConstView>(stream);
  view->moduleStart_ = moduleStart_d.get();
  view->clus_ = clus_d.get();
  view->clusInModule_ = clusInModule_d.get();
  view->moduleId_ = moduleId_d.get();
  view->clusModuleStart_ = clusModuleStart_d.get();

  view_d = cs->make_device_unique<DeviceConstView>(stream);
  cudaMemcpyAsync(view_d.get(), view.get(), sizeof(DeviceConstView), cudaMemcpyDefault, stream.id());
}
