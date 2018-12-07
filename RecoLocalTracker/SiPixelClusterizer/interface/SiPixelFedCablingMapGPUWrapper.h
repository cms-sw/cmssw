#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "CUDADataFormats/Common/interface/device_unique_ptr.h"
#include "CUDADataFormats/Common/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAESProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelFedCablingMapGPU.h"

#include <cuda/api_wrappers.h>

#include <set>

class SiPixelFedCablingMap;
class TrackerGeometry;
class SiPixelQuality;

// TODO: since this has more information than just cabling map, maybe we should invent a better name?
class SiPixelFedCablingMapGPUWrapper {
public:
  SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMap const& cablingMap,
                                 TrackerGeometry const& trackerGeom,
                                 SiPixelQuality const *badPixelInfo);
  ~SiPixelFedCablingMapGPUWrapper();

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelFedCablingMapGPU *getGPUProductAsync(cuda::stream_t<>& cudaStream) const;

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAllAsync(cuda::stream_t<>& cudaStream) const;
  edm::cuda::device::unique_ptr<unsigned char[]> getModToUnpRegionalAsync(std::set<unsigned int> const& modules, cuda::stream_t<>& cudaStream) const;

private:
  const SiPixelFedCablingMap *cablingMap_;
  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  fedMap;
  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  linkMap;
  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  rocMap;
  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  RawId;
  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  rocInDet;
  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  moduleId;
  std::vector<unsigned char, CUDAHostAllocator<unsigned char>> badRocs;
  std::vector<unsigned char, CUDAHostAllocator<unsigned char>> modToUnpDefault;
  unsigned int size;
  bool hasQuality_;

  struct GPUData {
    ~GPUData();
    SiPixelFedCablingMapGPU *cablingMapHost = nullptr;   // internal pointers are to GPU, struct itself is on CPU
    SiPixelFedCablingMapGPU *cablingMapDevice = nullptr; // same internal pointers as above, struct itself is on GPU
  };
  CUDAESProduct<GPUData> gpuData_;

  struct ModulesToUnpack {
    ~ModulesToUnpack();
    unsigned char *modToUnpDefault = nullptr; // pointer to GPU
  };
  CUDAESProduct<ModulesToUnpack> modToUnp_;
};


#endif
