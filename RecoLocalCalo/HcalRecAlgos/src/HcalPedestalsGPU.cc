#include "RecoLocalCalo/HcalRecAlgos/interface/HcalPedestalsGPU.h"

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// FIXME: add proper getters to conditions
HcalPedestalsGPU::HcalPedestalsGPU(HcalPedestals const& pedestals)
    : unitIsADC_{pedestals.isADC()},
      totalChannels_{pedestals.getAllContainers()[0].second.size() + pedestals.getAllContainers()[1].second.size()},
      offsetForHashes_{static_cast<uint32_t>(pedestals.getAllContainers()[0].second.size())},
      values_(totalChannels_ * 4),
      widths_(totalChannels_ * 4) {
#ifdef HCAL_MAHI_CPUDEBUG
  std::cout << "unitIsADC = " << unitIsADC_ << std::endl;
#endif

  auto const containers = pedestals.getAllContainers();

  // fill in eb
  auto const& barrelValues = containers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    values_[i * 4] = barrelValues[i].getValue(0);
    values_[i * 4 + 1] = barrelValues[i].getValue(1);
    values_[i * 4 + 2] = barrelValues[i].getValue(2);
    values_[i * 4 + 3] = barrelValues[i].getValue(3);

    widths_[i * 4] = barrelValues[i].getWidth(0);
    widths_[i * 4 + 1] = barrelValues[i].getWidth(1);
    widths_[i * 4 + 2] = barrelValues[i].getWidth(2);
    widths_[i * 4 + 3] = barrelValues[i].getWidth(3);
  }

  // fill in ee
  auto const& endcapValues = containers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    auto const off = offset + i;
    values_[off * 4] = endcapValues[i].getValue(0);
    values_[off * 4 + 1] = endcapValues[i].getValue(1);
    values_[off * 4 + 2] = endcapValues[i].getValue(2);
    values_[off * 4 + 3] = endcapValues[i].getValue(3);

    widths_[off * 4] = endcapValues[i].getWidth(0);
    widths_[off * 4 + 1] = endcapValues[i].getWidth(1);
    widths_[off * 4 + 2] = endcapValues[i].getWidth(2);
    widths_[off * 4 + 3] = endcapValues[i].getWidth(3);
  }
}

HcalPedestalsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(values));
  cudaCheck(cudaFree(widths));
}

HcalPedestalsGPU::Product const& HcalPedestalsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalPedestalsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.values, this->values_.size() * sizeof(float)));

        cudaCheck(cudaMalloc((void**)&product.widths, this->widths_.size() * sizeof(float)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.values,
                                  this->values_.data(),
                                  this->values_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));

        cudaCheck(cudaMemcpyAsync(product.widths,
                                  this->widths_.data(),
                                  this->widths_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalPedestalsGPU);
