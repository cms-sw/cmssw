#ifndef RecoLocalTracker_SiStripClusterizer_plugins_SiStripRawToClusterGPUKernel_h
#define RecoLocalTracker_SiStripClusterizer_plugins_SiStripRawToClusterGPUKernel_h

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsGPU.h"
//#include "clusterGPU.cuh"

#include <cuda_runtime.h>

#include <vector>
#include <memory>

class ChannelLocs;
class ChannelLocsGPU;
class FEDRawData;

namespace sistrip {
  class FEDBuffer;
}
namespace edm {
  class ParameterSet;
}

namespace stripgpu {
  struct StripDataView;

  class StripDataGPU {
  public:
    StripDataGPU(size_t size, cudaStream_t stream);

    cms::cuda::device::unique_ptr<uint8_t[]> alldataGPU_;
    cms::cuda::device::unique_ptr<uint16_t[]> channelGPU_;
    cms::cuda::device::unique_ptr<stripgpu::stripId_t[]> stripIdGPU_;
    cms::cuda::device::unique_ptr<int[]> seedStripsMask_;
    cms::cuda::device::unique_ptr<int[]> prefixSeedStripsNCMask_;
  };

  class SiStripRawToClusterGPUKernel {
  public:
    SiStripRawToClusterGPUKernel(const edm::ParameterSet& conf);
    void makeAsync(const std::vector<const FEDRawData*>& rawdata,
                   const std::vector<std::unique_ptr<sistrip::FEDBuffer>>& buffers,
                   const SiStripClusterizerConditionsGPU& conditions,
                   cudaStream_t stream);
    void copyAsync(cudaStream_t stream);
    SiStripClustersCUDADevice getResults(cudaStream_t stream);

  private:
    using ConditionsDeviceView = SiStripClusterizerConditionsGPU::Data::DeviceView;

    void reset();
    void unpackChannelsGPU(const ConditionsDeviceView* conditions, cudaStream_t stream);
    void allocateSSTDataGPU(int max_strips, cudaStream_t stream);
    void freeSSTDataGPU(cudaStream_t stream);

    void setSeedStripsNCIndexGPU(const ConditionsDeviceView* conditions, cudaStream_t stream);
    void findClusterGPU(const ConditionsDeviceView* conditions, cudaStream_t stream);

    std::vector<stripgpu::fedId_t> fedIndex_;
    std::vector<size_t> fedRawDataOffsets_;

    std::unique_ptr<StripDataGPU> stripdata_;
    std::unique_ptr<ChannelLocsGPU> chanlocsGPU_;

    cms::cuda::host::unique_ptr<StripDataView> sst_data_d_;
    cms::cuda::device::unique_ptr<StripDataView> pt_sst_data_d_;

    SiStripClustersCUDADevice clusters_d_;
    float channelThreshold_, seedThreshold_, clusterThresholdSquared_;
    uint8_t maxSequentialHoles_, maxSequentialBad_, maxAdjacentBad_;
    uint32_t maxClusterSize_;
    float minGoodCharge_;
  };
}  // namespace stripgpu
#endif
