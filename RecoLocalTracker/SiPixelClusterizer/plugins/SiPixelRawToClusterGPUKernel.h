#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h

#include <algorithm>
#include <cuda_runtime.h>

#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigiErrorsCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "FWCore/Utilities/interface/typedefs.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

// #define GPU_DEBUG

// local include(s)
#include "SiPixelClusterThresholds.h"

struct SiPixelROCsStatusAndMapping;
class SiPixelGainForHLTonGPU;

namespace pixelgpudetails {

  inline namespace phase1geometry {
    const uint32_t layerStartBit = 20;
    const uint32_t ladderStartBit = 12;
    const uint32_t moduleStartBit = 2;

    const uint32_t panelStartBit = 10;
    const uint32_t diskStartBit = 18;
    const uint32_t bladeStartBit = 12;

    const uint32_t layerMask = 0xF;
    const uint32_t ladderMask = 0xFF;
    const uint32_t moduleMask = 0x3FF;
    const uint32_t panelMask = 0x3;
    const uint32_t diskMask = 0xF;
    const uint32_t bladeMask = 0x3F;
  }  // namespace phase1geometry

  const uint32_t maxROCIndex = 8;
  const uint32_t numRowsInRoc = 80;
  const uint32_t numColsInRoc = 52;

  const uint32_t MAX_WORD = 2000;

  struct DetIdGPU {
    uint32_t rawId;
    uint32_t rocInDet;
    uint32_t moduleId;
  };

  struct Pixel {
    uint32_t row;
    uint32_t col;
  };

  inline constexpr pixelchannelidentifierimpl::Packing packing() { return PixelChannelIdentifier::thePacking; }

  inline constexpr uint32_t pack(uint32_t row, uint32_t col, uint32_t adc, uint32_t flag = 0) {
    constexpr pixelchannelidentifierimpl::Packing thePacking = packing();
    adc = std::min(adc, uint32_t(thePacking.max_adc));

    return (row << thePacking.row_shift) | (col << thePacking.column_shift) | (adc << thePacking.adc_shift);
  }

  constexpr uint32_t pixelToChannel(int row, int col) {
    constexpr pixelchannelidentifierimpl::Packing thePacking = packing();
    return (row << thePacking.column_width) | col;
  }

  template <typename TrackerTraits>
  class SiPixelRawToClusterGPUKernel {
  public:
    class WordFedAppender {
    public:
      WordFedAppender(uint32_t words, cudaStream_t stream)
          : word_{cms::cuda::make_host_unique<unsigned int[]>(words, stream)},
            fedId_{cms::cuda::make_host_unique<unsigned char[]>(words, stream)} {}

      void initializeWordFed(int fedId, unsigned int index, cms_uint32_t const* src, unsigned int length) {
        std::memcpy(word_.get() + index, src, sizeof(cms_uint32_t) * length);
        std::memset(fedId_.get() + index / 2, fedId - FEDNumbering::MINSiPixeluTCAFEDID, length / 2);
      }

      const unsigned int* word() const { return word_.get(); }
      const unsigned char* fedId() const { return fedId_.get(); }

    private:
      cms::cuda::host::unique_ptr<unsigned int[]> word_;
      cms::cuda::host::unique_ptr<unsigned char[]> fedId_;
    };

    SiPixelRawToClusterGPUKernel() = default;
    ~SiPixelRawToClusterGPUKernel() = default;

    SiPixelRawToClusterGPUKernel(const SiPixelRawToClusterGPUKernel&) = delete;
    SiPixelRawToClusterGPUKernel(SiPixelRawToClusterGPUKernel&&) = delete;
    SiPixelRawToClusterGPUKernel& operator=(const SiPixelRawToClusterGPUKernel&) = delete;
    SiPixelRawToClusterGPUKernel& operator=(SiPixelRawToClusterGPUKernel&&) = delete;

    void makePhase1ClustersAsync(const SiPixelClusterThresholds clusterThresholds,
                                 const SiPixelROCsStatusAndMapping* cablingMap,
                                 const unsigned char* modToUnp,
                                 const SiPixelGainForHLTonGPU* gains,
                                 const WordFedAppender& wordFed,
                                 SiPixelFormatterErrors&& errors,
                                 const uint32_t wordCounter,
                                 const uint32_t fedCounter,
                                 bool useQualityInfo,
                                 bool includeErrors,
                                 bool debug,
                                 cudaStream_t stream);

    void makePhase2ClustersAsync(const SiPixelClusterThresholds clusterThresholds,
                                 const uint16_t* moduleIds,
                                 const uint16_t* xDigis,
                                 const uint16_t* yDigis,
                                 const uint16_t* adcDigis,
                                 const uint32_t* packedData,
                                 const uint32_t* rawIds,
                                 const uint32_t numDigis,
                                 cudaStream_t stream);

    std::pair<SiPixelDigisCUDA, SiPixelClustersCUDA> getResults() {
      digis_d.setNModulesDigis(nModules_Clusters_h[0], nDigis);
      assert(nModules_Clusters_h[2] <= nModules_Clusters_h[1]);
      clusters_d.setNClusters(nModules_Clusters_h[1], nModules_Clusters_h[2]);
      // need to explicitly deallocate while the associated CUDA
      // stream is still alive
      //
      // technically the statement above is not true anymore now that
      // the CUDA streams are cached within the cms::cuda::StreamCache, but it is
      // still better to release as early as possible
      nModules_Clusters_h.reset();
      return std::make_pair(std::move(digis_d), std::move(clusters_d));
    }

    SiPixelDigiErrorsCUDA&& getErrors() { return std::move(digiErrors_d); }

  private:
    uint32_t nDigis;

    // Data to be put in the event
    cms::cuda::host::unique_ptr<uint32_t[]> nModules_Clusters_h;
    SiPixelDigisCUDA digis_d;
    SiPixelClustersCUDA clusters_d;
    SiPixelDigiErrorsCUDA digiErrors_d;
  };

}  // namespace pixelgpudetails

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h
