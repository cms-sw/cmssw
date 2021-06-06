#ifndef EventFilter_HcalRawToDigi_interface_DeclsForKernels_h
#define EventFilter_HcalRawToDigi_interface_DeclsForKernels_h

#include <vector>

#include "CUDADataFormats/HcalDigi/interface/DigiCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "ElectronicsMappingGPU.h"

namespace hcal {
  namespace raw {

    constexpr int32_t empty_event_size = 32;
    constexpr uint32_t utca_nfeds_max = 50;
    constexpr uint32_t nbytes_per_fed_max = 10 * 1024;

    // each collection corresponds to a particular flavor with a certain number of
    // samples per digi
    constexpr uint32_t numOutputCollections = 3;
    constexpr uint8_t OutputF01HE = 0;
    constexpr uint8_t OutputF5HB = 1;
    constexpr uint8_t OutputF3HB = 2;

    struct ConfigurationParameters {
      uint32_t maxChannelsF01HE;
      uint32_t maxChannelsF5HB;
      uint32_t maxChannelsF3HB;
      uint32_t nsamplesF01HE;
      uint32_t nsamplesF5HB;
      uint32_t nsamplesF3HB;
    };

    struct InputDataCPU {
      cms::cuda::host::unique_ptr<unsigned char[]> data;
      cms::cuda::host::unique_ptr<uint32_t[]> offsets;
      cms::cuda::host::unique_ptr<int[]> feds;
    };

    struct OutputDataCPU {
      cms::cuda::host::unique_ptr<uint32_t[]> nchannels;
    };

    struct ScratchDataGPU {
      // depends on the number of output collections
      // that is a statically known predefined number
      cms::cuda::device::unique_ptr<uint32_t[]> pChannelsCounters;
    };

    struct OutputDataGPU {
      DigiCollection<Flavor1, ::calo::common::DevStoragePolicy> digisF01HE;
      DigiCollection<Flavor5, ::calo::common::DevStoragePolicy> digisF5HB;
      DigiCollection<Flavor3, ::calo::common::DevStoragePolicy> digisF3HB;

      void allocate(ConfigurationParameters const &config, cudaStream_t cudaStream) {
        digisF01HE.data = cms::cuda::make_device_unique<uint16_t[]>(
            config.maxChannelsF01HE * compute_stride<Flavor1>(config.nsamplesF01HE), cudaStream);
        digisF01HE.ids = cms::cuda::make_device_unique<uint32_t[]>(config.maxChannelsF01HE, cudaStream);

        digisF5HB.data = cms::cuda::make_device_unique<uint16_t[]>(
            config.maxChannelsF5HB * compute_stride<Flavor5>(config.nsamplesF5HB), cudaStream);
        digisF5HB.ids = cms::cuda::make_device_unique<uint32_t[]>(config.maxChannelsF5HB, cudaStream);
        digisF5HB.npresamples = cms::cuda::make_device_unique<uint8_t[]>(config.maxChannelsF5HB, cudaStream);

        digisF3HB.data = cms::cuda::make_device_unique<uint16_t[]>(
            config.maxChannelsF3HB * compute_stride<Flavor3>(config.nsamplesF3HB), cudaStream);
        digisF3HB.ids = cms::cuda::make_device_unique<uint32_t[]>(config.maxChannelsF3HB, cudaStream);
      }
    };

    struct InputDataGPU {
      cms::cuda::device::unique_ptr<unsigned char[]> data;
      cms::cuda::device::unique_ptr<uint32_t[]> offsets;
      cms::cuda::device::unique_ptr<int[]> feds;
    };

    struct ConditionsProducts {
      ElectronicsMappingGPU::Product const &eMappingProduct;
    };

  }  // namespace raw
}  // namespace hcal

#endif  // EventFilter_HcalRawToDigi_interface_DeclsForKernels_h
