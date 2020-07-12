#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelFedCablingMapGPU.h"

#include <cuda_runtime.h>

#include <set>

class SiPixelFedCablingMap;
class TrackerGeometry;
class SiPixelQuality;

// TODO: since this has more information than just cabling map, maybe we should invent a better name?
class SiPixelFedCablingMapGPUWrapper {
public:
  SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMap const &cablingMap,
                                 TrackerGeometry const &trackerGeom,
                                 SiPixelQuality const *badPixelInfo);
  ~SiPixelFedCablingMapGPUWrapper();

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelFedCablingMapGPU *getGPUProductAsync(cudaStream_t cudaStream) const;

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAllAsync(cudaStream_t cudaStream) const;
  cms::cuda::device::unique_ptr<unsigned char[]> getModToUnpRegionalAsync(std::set<unsigned int> const &modules,
                                                                          cudaStream_t cudaStream) const;

private:
  const SiPixelFedCablingMap *cablingMap_;
  std::vector<unsigned char, cms::cuda::HostAllocator<unsigned char>> modToUnpDefault;
  unsigned int size;
  bool hasQuality_;

  SiPixelFedCablingMapGPU *cablingMapHost = nullptr;  // pointer to struct in CPU

  struct GPUData {
    ~GPUData();
    SiPixelFedCablingMapGPU *cablingMapDevice = nullptr;  // pointer to struct in GPU
  };
  cms::cuda::ESProduct<GPUData> gpuData_;

  struct ModulesToUnpack {
    ~ModulesToUnpack();
    unsigned char *modToUnpDefault = nullptr;  // pointer to GPU
  };
  cms::cuda::ESProduct<ModulesToUnpack> modToUnp_;
};

#endif
