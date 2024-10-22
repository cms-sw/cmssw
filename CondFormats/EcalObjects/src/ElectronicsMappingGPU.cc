#include "CondFormats/EcalObjects/interface/ElectronicsMappingGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"

namespace ecal {
  namespace raw {

    // TODO: 0x3FFFFF * 4B ~= 16MB
    // tmp solution for linear mapping of eid -> did
    ElectronicsMappingGPU::ElectronicsMappingGPU(EcalMappingElectronics const& mapping) : eid2did_(0x3FFFFF) {
      // fill in eb
      // TODO: EB vector is actually empty
      auto const& barrelValues = mapping.barrelItems();
      for (unsigned int i = 0; i < barrelValues.size(); i++) {
        EcalElectronicsId eid{barrelValues[i].electronicsid};
        EBDetId did{EBDetId::unhashIndex(i)};
        eid2did_[eid.linearIndex()] = did.rawId();
      }

      // fill in ee
      auto const& endcapValues = mapping.endcapItems();
      for (unsigned int i = 0; i < endcapValues.size(); i++) {
        EcalElectronicsId eid{endcapValues[i].electronicsid};
        EEDetId did{EEDetId::unhashIndex(i)};
        eid2did_[eid.linearIndex()] = did.rawId();
      }
    }

    ElectronicsMappingGPU::Product const& ElectronicsMappingGPU::getProduct(cudaStream_t cudaStream) const {
      auto const& product = product_.dataForCurrentDeviceAsync(
          cudaStream, [this](ElectronicsMappingGPU::Product& product, cudaStream_t cudaStream) {
            // allocate
            product.eid2did = cms::cuda::make_device_unique<uint32_t[]>(eid2did_.size(), cudaStream);
            // transfer
            cms::cuda::copyAsync(product.eid2did, eid2did_, cudaStream);
          });

      return product;
    }

  }  // namespace raw
}  // namespace ecal

TYPELOOKUP_DATA_REG(ecal::raw::ElectronicsMappingGPU);
