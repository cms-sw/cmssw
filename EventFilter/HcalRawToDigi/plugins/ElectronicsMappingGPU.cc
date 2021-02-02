#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "ElectronicsMappingGPU.h"

namespace hcal {
  namespace raw {

    // TODO: 0x3FFFFF * 4B ~= 16MB
    // tmp solution for linear mapping of eid -> did
    ElectronicsMappingGPU::ElectronicsMappingGPU(HcalElectronicsMap const& mapping)
        : eid2tid_(HcalElectronicsId::maxLinearIndex, 0u), eid2did_(HcalElectronicsId::maxLinearIndex, 0u) {
      auto const& eidsPrecision = mapping.allElectronicsIdPrecision();
      for (uint32_t i = 0; i < eidsPrecision.size(); ++i) {
        auto const& eid = eidsPrecision[i];

        // assign
        eid2did_[eid.linearIndex()] = eid.isTriggerChainId() ? 0u : mapping.lookup(eid).rawId();
      }

      auto const& eidsTrigger = mapping.allElectronicsIdTrigger();
      for (uint32_t i = 0; i < eidsTrigger.size(); i++) {
        auto const& eid = eidsTrigger[i];

        // assign
        eid2tid_[eid.linearIndex()] = eid.isTriggerChainId() ? mapping.lookupTrigger(eid).rawId() : 0u;
      }
    }

    ElectronicsMappingGPU::Product::~Product() {
      // deallocation
      cudaCheck(cudaFree(eid2did));
      cudaCheck(cudaFree(eid2tid));
    }

    ElectronicsMappingGPU::Product const& ElectronicsMappingGPU::getProduct(cudaStream_t cudaStream) const {
      auto const& product = product_.dataForCurrentDeviceAsync(
          cudaStream, [this](ElectronicsMappingGPU::Product& product, cudaStream_t cudaStream) {
            // malloc
            cudaCheck(cudaMalloc((void**)&product.eid2did, this->eid2did_.size() * sizeof(uint32_t)));
            cudaCheck(cudaMalloc((void**)&product.eid2tid, this->eid2tid_.size() * sizeof(uint32_t)));

            // transfer
            cudaCheck(cudaMemcpyAsync(product.eid2did,
                                      this->eid2did_.data(),
                                      this->eid2did_.size() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
            cudaCheck(cudaMemcpyAsync(product.eid2tid,
                                      this->eid2tid_.data(),
                                      this->eid2tid_.size() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
          });

      return product;
    }

  }  // namespace raw
}  // namespace hcal

TYPELOOKUP_DATA_REG(hcal::raw::ElectronicsMappingGPU);
