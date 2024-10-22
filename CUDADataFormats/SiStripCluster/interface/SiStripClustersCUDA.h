#ifndef CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDA_h
#define CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDA_h

#include "DataFormats/SiStripCluster/interface/SiStripClustersSOABase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include <cuda_runtime.h>

class SiStripClustersCUDADevice : public SiStripClustersSOABase<cms::cuda::device::unique_ptr> {
public:
  SiStripClustersCUDADevice() = default;
  explicit SiStripClustersCUDADevice(uint32_t maxClusters, uint32_t maxStripsPerCluster, cudaStream_t stream);
  ~SiStripClustersCUDADevice() override = default;

  SiStripClustersCUDADevice(const SiStripClustersCUDADevice &) = delete;
  SiStripClustersCUDADevice &operator=(const SiStripClustersCUDADevice &) = delete;
  SiStripClustersCUDADevice(SiStripClustersCUDADevice &&) = default;
  SiStripClustersCUDADevice &operator=(SiStripClustersCUDADevice &&) = default;

  struct DeviceView {
    uint32_t *clusterIndex_;
    uint32_t *clusterSize_;
    uint8_t *clusterADCs_;
    stripgpu::detId_t *clusterDetId_;
    stripgpu::stripId_t *firstStrip_;
    bool *trueCluster_;
    float *barycenter_;
    float *charge_;
    uint32_t nClusters_;
    uint32_t maxClusterSize_;
  };

  DeviceView *view() const { return view_d.get(); }
  uint32_t nClusters() const { return nClusters_; }
  uint32_t *nClustersPtr() { return &nClusters_; }
  uint32_t maxClusterSize() const { return maxClusterSize_; }
  uint32_t *maxClusterSizePtr() { return &maxClusterSize_; }

private:
  cms::cuda::device::unique_ptr<DeviceView> view_d;  // "me" pointer
  uint32_t nClusters_;
  uint32_t maxClusterSize_;
};

class SiStripClustersCUDAHost : public SiStripClustersSOABase<cms::cuda::host::unique_ptr> {
public:
  SiStripClustersCUDAHost() = default;
  explicit SiStripClustersCUDAHost(const SiStripClustersCUDADevice &clusters_d, cudaStream_t stream);
  ~SiStripClustersCUDAHost() override = default;

  SiStripClustersCUDAHost(const SiStripClustersCUDAHost &) = delete;
  SiStripClustersCUDAHost &operator=(const SiStripClustersCUDAHost &) = delete;
  SiStripClustersCUDAHost(SiStripClustersCUDAHost &&) = default;
  SiStripClustersCUDAHost &operator=(SiStripClustersCUDAHost &&) = default;
};

#endif
