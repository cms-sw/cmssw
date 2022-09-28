#ifndef EventFilter_EcalRawToDigi_plugins_DeclsForKernels_h
#define EventFilter_EcalRawToDigi_plugins_DeclsForKernels_h

#include <vector>

#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"
#include "CondFormats/EcalObjects/interface/ElectronicsMappingGPU.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "EventFilter/EcalRawToDigi/interface/DCCRawDataDefinitions.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

namespace ecal {
  namespace raw {

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
        memoryPool::Deleter deleter = memoryPool::Deleter(std::make_shared<memoryPool::cuda::BundleDelete>(cudaStream, memoryPool::onDevice));
        assert(deleter.pool());
        digisEB.data =
            memoryPool::cuda::makeBuffer<uint16_t>(config.maxChannelsEB * EcalDataFrame::MAXSAMPLES, deleter);
        digisEE.data =
            memoryPool::cuda::makeBuffer<uint16_t>(config.maxChannelsEE * EcalDataFrame::MAXSAMPLES, deleter);
        digisEB.ids = memoryPool::cuda::makeBuffer<uint32_t>(config.maxChannelsEB, deleter);
        digisEE.ids = memoryPool::cuda::makeBuffer<uint32_t>(config.maxChannelsEE, deleter);
      }
    };

    struct ScratchDataGPU {
      // [0] = EB
      // [1] = EE
      memoryPool::Buffer<uint32_t> pChannelsCounter;
    };

    struct InputDataGPU {
      memoryPool::Buffer<unsigned char> data;
      memoryPool::Buffer<uint32_t> offsets;
      memoryPool::Buffer<int> feds;
    };

    struct ConditionsProducts {
      ElectronicsMappingGPU::Product const &eMappingProduct;
    };

  }  // namespace raw
}  // namespace ecal

#endif  // EventFilter_EcalRawToDigi_plugins_DeclsForKernels_h
