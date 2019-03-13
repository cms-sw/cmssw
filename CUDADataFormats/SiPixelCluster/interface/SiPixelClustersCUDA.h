#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include <cuda/api_wrappers.h>

class SiPixelClustersCUDA {
public:
  SiPixelClustersCUDA() = default;
  explicit SiPixelClustersCUDA(size_t maxClusters, cuda::stream_t<>& stream);
  ~SiPixelClustersCUDA() = default;

  SiPixelClustersCUDA(const SiPixelClustersCUDA&) = delete;
  SiPixelClustersCUDA& operator=(const SiPixelClustersCUDA&) = delete;
  SiPixelClustersCUDA(SiPixelClustersCUDA&&) = default;
  SiPixelClustersCUDA& operator=(SiPixelClustersCUDA&&) = default;

  void setNClusters(uint32_t nClusters) {
    nClusters_h = nClusters;
  }

  uint32_t nClusters() const { return nClusters_h; }

  uint32_t *moduleStart() { return moduleStart_d.get(); }
  uint32_t *clusInModule() { return clusInModule_d.get(); }
  uint32_t *moduleId() { return moduleId_d.get(); }
  uint32_t *clusModuleStart() { return clusModuleStart_d.get(); }

  uint32_t const *moduleStart() const { return moduleStart_d.get(); }
  uint32_t const *clusInModule() const { return clusInModule_d.get(); }
  uint32_t const *moduleId() const { return moduleId_d.get(); }
  uint32_t const *clusModuleStart() const { return clusModuleStart_d.get(); }

  uint32_t const *c_moduleStart() const { return moduleStart_d.get(); }
  uint32_t const *c_clusInModule() const { return clusInModule_d.get(); }
  uint32_t const *c_moduleId() const { return moduleId_d.get(); }
  uint32_t const *c_clusModuleStart() const { return clusModuleStart_d.get(); }

  class DeviceConstView {
  public:
    DeviceConstView() = default;

#ifdef __CUDACC__
    __device__ __forceinline__ uint32_t moduleStart(int i) const { return __ldg(moduleStart_+i); }
    __device__ __forceinline__ uint32_t clusInModule(int i) const { return __ldg(clusInModule_+i); }
    __device__ __forceinline__ uint32_t moduleId(int i) const { return __ldg(moduleId_+i); }
    __device__ __forceinline__ uint32_t clusModuleStart(int i) const { return __ldg(clusModuleStart_+i); }
#endif

    friend SiPixelClustersCUDA;

  private:
    uint32_t const *moduleStart_;
    uint32_t const *clusInModule_;
    uint32_t const *moduleId_;
    uint32_t const *clusModuleStart_;
  };

  DeviceConstView *view() const { return view_d.get(); }

private:
  cudautils::device::unique_ptr<uint32_t[]> moduleStart_d;   // index of the first pixel of each module
  cudautils::device::unique_ptr<uint32_t[]> clusInModule_d;  // number of clusters found in each module
  cudautils::device::unique_ptr<uint32_t[]> moduleId_d;      // module id of each module

  // originally from rechits
  cudautils::device::unique_ptr<uint32_t[]> clusModuleStart_d;

  cudautils::device::unique_ptr<DeviceConstView> view_d;    // "me" pointer

  uint32_t nClusters_h;
};

#endif
