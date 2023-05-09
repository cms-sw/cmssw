#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

SiStripClustersCUDADevice::SiStripClustersCUDADevice(uint32_t maxClusters,
                                                     uint32_t maxStripsPerCluster,
                                                     cudaStream_t stream) {
  maxClusterSize_ = maxStripsPerCluster;

  clusterIndex_ = cms::cuda::make_device_unique<uint32_t[]>(maxClusters, stream);
  clusterSize_ = cms::cuda::make_device_unique<uint32_t[]>(maxClusters, stream);
  clusterADCs_ = cms::cuda::make_device_unique<uint8_t[]>(maxClusters * maxStripsPerCluster, stream);
  clusterDetId_ = cms::cuda::make_device_unique<stripgpu::detId_t[]>(maxClusters, stream);
  firstStrip_ = cms::cuda::make_device_unique<stripgpu::stripId_t[]>(maxClusters, stream);
  trueCluster_ = cms::cuda::make_device_unique<bool[]>(maxClusters, stream);
  barycenter_ = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  charge_ = cms::cuda::make_device_unique<float[]>(maxClusters, stream);

  auto view = cms::cuda::make_host_unique<DeviceView>(stream);
  view->clusterIndex_ = clusterIndex_.get();
  view->clusterSize_ = clusterSize_.get();
  view->clusterADCs_ = clusterADCs_.get();
  view->clusterDetId_ = clusterDetId_.get();
  view->firstStrip_ = firstStrip_.get();
  view->trueCluster_ = trueCluster_.get();
  view->barycenter_ = barycenter_.get();
  view->charge_ = charge_.get();
  view->maxClusterSize_ = maxStripsPerCluster;

  view_d = cms::cuda::make_device_unique<DeviceView>(stream);
  cms::cuda::copyAsync(view_d, view, stream);
#ifdef GPU_CHECK
  cudaCheck(cudaStreamSynchronize(stream));
#endif
}

SiStripClustersCUDAHost::SiStripClustersCUDAHost(const SiStripClustersCUDADevice& clusters_d, cudaStream_t stream) {
  nClusters_ = clusters_d.nClusters();
  maxClusterSize_ = clusters_d.maxClusterSize();
  clusterIndex_ = cms::cuda::make_host_unique<uint32_t[]>(nClusters_, stream);
  clusterSize_ = cms::cuda::make_host_unique<uint32_t[]>(nClusters_, stream);
  clusterADCs_ = cms::cuda::make_host_unique<uint8_t[]>(nClusters_ * maxClusterSize_, stream);
  clusterDetId_ = cms::cuda::make_host_unique<stripgpu::detId_t[]>(nClusters_, stream);
  firstStrip_ = cms::cuda::make_host_unique<stripgpu::stripId_t[]>(nClusters_, stream);
  trueCluster_ = cms::cuda::make_host_unique<bool[]>(nClusters_, stream);
  barycenter_ = cms::cuda::make_host_unique<float[]>(nClusters_, stream);
  charge_ = cms::cuda::make_host_unique<float[]>(nClusters_, stream);

  cms::cuda::copyAsync(clusterIndex_, clusters_d.clusterIndex(), nClusters_, stream);
  cms::cuda::copyAsync(clusterSize_, clusters_d.clusterSize(), nClusters_, stream);
  cms::cuda::copyAsync(clusterADCs_, clusters_d.clusterADCs(), nClusters_ * maxClusterSize_, stream);
  cms::cuda::copyAsync(clusterDetId_, clusters_d.clusterDetId(), nClusters_, stream);
  cms::cuda::copyAsync(firstStrip_, clusters_d.firstStrip(), nClusters_, stream);
  cms::cuda::copyAsync(trueCluster_, clusters_d.trueCluster(), nClusters_, stream);
  cms::cuda::copyAsync(barycenter_, clusters_d.barycenter(), nClusters_, stream);
  cms::cuda::copyAsync(charge_, clusters_d.charge(), nClusters_, stream);
#ifdef GPU_CHECK
  cudaCheck(cudaStreamSynchronize(stream));
#endif
}
