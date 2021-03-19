#ifndef EventFilter_EcalRawToDigi_plugins_DeclsForKernels_h
#define EventFilter_EcalRawToDigi_plugins_DeclsForKernels_h

#include <vector>

#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "EventFilter/EcalRawToDigi/interface/DCCRawDataDefinitions.h"
#include "EventFilter/EcalRawToDigi/interface/ElectronicsMappingGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

namespace ecal {
  namespace raw {

    constexpr auto empty_event_size = EMPTYEVENTSIZE;
    constexpr uint32_t nfeds_max = 54;
    constexpr uint32_t nbytes_per_fed_max = 10 * 1024;

    struct InputDataCPU {
      cms::cuda::host::unique_ptr<unsigned char[]> data;
      cms::cuda::host::unique_ptr<uint32_t[]> offsets;
      cms::cuda::host::unique_ptr<int[]> feds;
    };

    struct ConfigurationParameters {
      uint32_t maxChannelsEE;
      uint32_t maxChannelsEB;
    };

    struct OutputDataCPU {
      // [0] - eb, [1] - ee
      cms::cuda::host::unique_ptr<uint32_t[]> nchannels;
    };

    struct OutputDataGPU {
      DigisCollection<::calo::common::DevStoragePolicy> digisEB, digisEE;

      void allocate(ConfigurationParameters const &config, cudaStream_t cudaStream) {
        digisEB.data =
            cms::cuda::make_device_unique<uint16_t[]>(config.maxChannelsEB * EcalDataFrame::MAXSAMPLES, cudaStream);
        digisEE.data =
            cms::cuda::make_device_unique<uint16_t[]>(config.maxChannelsEE * EcalDataFrame::MAXSAMPLES, cudaStream);
        digisEB.ids = cms::cuda::make_device_unique<uint32_t[]>(config.maxChannelsEB, cudaStream);
        digisEE.ids = cms::cuda::make_device_unique<uint32_t[]>(config.maxChannelsEE, cudaStream);
      }
    };

    struct ScratchDataGPU {
      // [0] = EB
      // [1] = EE
      cms::cuda::device::unique_ptr<uint32_t[]> pChannelsCounter;
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
}  // namespace ecal

#endif  // EventFilter_EcalRawToDigi_plugins_DeclsForKernels_h
