#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelROCsStatusAndMappingWrapper_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelROCsStatusAndMappingWrapper_h

#include <set>

#include <cuda_runtime.h>

#include "CondFormats/SiPixelObjects/interface/SiPixelROCsStatusAndMapping.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

class SiPixelFedCablingMap;
class TrackerGeometry;
class SiPixelQuality;

// TODO: since this has more information than just cabling map, maybe we should invent a better name?
class SiPixelROCsStatusAndMappingWrapper {
public:
  SiPixelROCsStatusAndMappingWrapper(SiPixelFedCablingMap const &cablingMap,
                                     TrackerGeometry const &trackerGeom,
                                     SiPixelQuality const *badPixelInfo);
  ~SiPixelROCsStatusAndMappingWrapper();

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelROCsStatusAndMapping *getGPUProductAsync(cudaStream_t cudaStream) const;

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAllAsync(cudaStream_t cudaStream) const;
  cms::cuda::device::unique_ptr<unsigned char[]> getModToUnpRegionalAsync(std::set<unsigned int> const &modules,
                                                                          cudaStream_t cudaStream) const;

private:
  const SiPixelFedCablingMap *cablingMap_;
  std::vector<unsigned char, cms::cuda::HostAllocator<unsigned char>> modToUnpDefault;
  unsigned int size;
  bool hasQuality_;

  SiPixelROCsStatusAndMapping *cablingMapHost = nullptr;  // pointer to struct in CPU

  struct GPUData {
    ~GPUData();
    SiPixelROCsStatusAndMapping *cablingMapDevice = nullptr;  // pointer to struct in GPU
  };
  cms::cuda::ESProduct<GPUData> gpuData_;

  struct ModulesToUnpack {
    ~ModulesToUnpack();
    unsigned char *modToUnpDefault = nullptr;  // pointer to GPU
  };
  cms::cuda::ESProduct<ModulesToUnpack> modToUnp_;
};

#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelROCsStatusAndMappingWrapper_h
