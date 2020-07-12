#include "RecoLocalCalo/HcalRecAlgos/interface/HcalQIECodersGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

HcalQIECodersGPU::HcalQIECodersGPU(HcalQIEData const& qiedata)
    : totalChannels_{qiedata.getAllContainers()[0].second.size() + qiedata.getAllContainers()[1].second.size()},
      offsets_(totalChannels_ * numValuesPerChannel),
      slopes_(totalChannels_ * numValuesPerChannel) {
  auto const containers = qiedata.getAllContainers();

  // fill in hb
  auto const& barrelValues = containers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    for (uint32_t k = 0; k < 4; k++)
      for (uint32_t l = 0; l < 4; l++) {
        auto const linear = k * 4 + l;
        offsets_[i * numValuesPerChannel + linear] = barrelValues[i].offset(k, l);
        slopes_[i * numValuesPerChannel + linear] = barrelValues[i].slope(k, l);
      }
  }

  // fill in he
  auto const& endcapValues = containers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    auto const off = (i + offset) * numValuesPerChannel;
    for (uint32_t k = 0; k < 4; k++)
      for (uint32_t l = 0; l < 4; l++) {
        auto const linear = k * 4u + l;
        offsets_[off + linear] = endcapValues[i].offset(k, l);
        slopes_[off + linear] = endcapValues[i].slope(k, l);
      }
  }
}

HcalQIECodersGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(offsets));
  cudaCheck(cudaFree(slopes));
}

HcalQIECodersGPU::Product const& HcalQIECodersGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalQIECodersGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.offsets, this->offsets_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.slopes, this->slopes_.size() * sizeof(float)));

        // transfer
        // offset
        cudaCheck(cudaMemcpyAsync(product.offsets,
                                  this->offsets_.data(),
                                  this->offsets_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));

        // slope
        cudaCheck(cudaMemcpyAsync(product.slopes,
                                  this->slopes_.data(),
                                  this->slopes_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalQIECodersGPU);
