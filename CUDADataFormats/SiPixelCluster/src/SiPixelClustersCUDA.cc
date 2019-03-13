#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t maxClusters, cuda::stream_t<>& stream) {
  edm::Service<CUDAService> cs;

  moduleStart_d     = cs->make_device_unique<uint32_t[]>(maxClusters+1, stream);
  clusInModule_d    = cs->make_device_unique<uint32_t[]>(maxClusters, stream);
  moduleId_d        = cs->make_device_unique<uint32_t[]>(maxClusters, stream);
  clusModuleStart_d = cs->make_device_unique<uint32_t[]>(maxClusters+1, stream);

  auto view = cs->make_host_unique<DeviceConstView>(stream);
  view->moduleStart_ = moduleStart_d.get();
  view->clusInModule_ = clusInModule_d.get();
  view->moduleId_ = moduleId_d.get();
  view->clusModuleStart_ = clusModuleStart_d.get();

  view_d = cs->make_device_unique<DeviceConstView>(stream);
  cudautils::copyAsync(view_d, view, stream);
}
