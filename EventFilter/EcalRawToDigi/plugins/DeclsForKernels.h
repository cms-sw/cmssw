#ifndef EventFilter_EcalRawToDigi_plugins_DeclsForKernels_h
#define EventFilter_EcalRawToDigi_plugins_DeclsForKernels_h

#include <vector>

#include "EventFilter/EcalRawToDigi/interface/DCCRawDataDefinitions.h"
#include "EventFilter/EcalRawToDigi/interface/ElectronicsMappingGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

namespace ecal {
  namespace raw {

    constexpr auto empty_event_size = EMPTYEVENTSIZE;
    constexpr uint32_t nfeds_max = 54;
    constexpr uint32_t nbytes_per_fed_max = 10 * 1024;

    struct InputDataCPU {
      std::vector<unsigned char, cms::cuda::HostAllocator<unsigned char>> data;
      std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> offsets;
      std::vector<int, cms::cuda::HostAllocator<int>> feds;

      void allocate() {
        // 2KB per FED resize
        data.resize(nfeds_max * sizeof(unsigned char) * nbytes_per_fed_max);
        offsets.resize(nfeds_max, 0);
        feds.resize(nfeds_max, 0);
      }
    };

    struct ConfigurationParameters {
      uint32_t maxChannels;
    };

    struct OutputDataCPU {
      // [0] - eb, [1] - ee
      std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> nchannels;

      void allocate() { nchannels.resize(2); }
    };

    struct OutputDataGPU {
      uint16_t *samplesEB = nullptr, *samplesEE = nullptr;
      uint32_t *idsEB = nullptr, *idsEE = nullptr;

      // FIXME: we should separate max channels parameter for eb and ee
      // FIXME: replace hardcoded values
      void allocate(ConfigurationParameters const &config) {
        cudaCheck(cudaMalloc((void **)&samplesEB, config.maxChannels * sizeof(uint16_t) * 10));
        cudaCheck(cudaMalloc((void **)&samplesEE, config.maxChannels * sizeof(uint16_t) * 10));
        cudaCheck(cudaMalloc((void **)&idsEB, config.maxChannels * sizeof(uint32_t)));
        cudaCheck(cudaMalloc((void **)&idsEE, config.maxChannels * sizeof(uint32_t)));
      }

      void deallocate(ConfigurationParameters const &config) {
        if (samplesEB) {
          cudaCheck(cudaFree(samplesEB));
          cudaCheck(cudaFree(samplesEE));
          cudaCheck(cudaFree(idsEB));
          cudaCheck(cudaFree(idsEE));
        }
      }
    };

    struct ScratchDataGPU {
      // [0] = EB
      // [1] = EE
      uint32_t *pChannelsCounter = nullptr;

      void allocate(ConfigurationParameters const &config) {
        cudaCheck(cudaMalloc((void **)&pChannelsCounter, sizeof(uint32_t) * 2));
      }

      void deallocate(ConfigurationParameters const &config) {
        if (pChannelsCounter) {
          cudaCheck(cudaFree(pChannelsCounter));
        }
      }
    };

    struct InputDataGPU {
      unsigned char *data = nullptr;
      uint32_t *offsets = nullptr;
      int *feds = nullptr;

      void allocate() {
        cudaCheck(cudaMalloc((void **)&data, sizeof(unsigned char) * nbytes_per_fed_max * nfeds_max));
        cudaCheck(cudaMalloc((void **)&offsets, sizeof(uint32_t) * nfeds_max));
        cudaCheck(cudaMalloc((void **)&feds, sizeof(int) * nfeds_max));
      }

      void deallocate() {
        if (data) {
          cudaCheck(cudaFree(data));
          cudaCheck(cudaFree(offsets));
          cudaCheck(cudaFree(feds));
        }
      }
    };

    struct ConditionsProducts {
      ElectronicsMappingGPU::Product const &eMappingProduct;
    };

  }  // namespace raw
}  // namespace ecal

#endif  // EventFilter_EcalRawToDigi_plugins_DeclsForKernels_h
