#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLinearCorrectionsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

EcalLinearCorrectionsGPU::EcalLinearCorrectionsGPU(EcalLinearCorrections const& values)
    : p1_(values.getValueMap().size()),
      p2_(values.getValueMap().size()),
      p3_(values.getValueMap().size()),
      t1_(values.getTimeMap().size()),
      t2_(values.getTimeMap().size()),
      t3_(values.getTimeMap().size()) {
  // fill in eb
  for (unsigned int i = 0; i < values.getValueMap().barrelItems().size(); i++) {
    p1_[i] = values.getValueMap().barrelItems()[i].p1;
    p2_[i] = values.getValueMap().barrelItems()[i].p2;
    p3_[i] = values.getValueMap().barrelItems()[i].p3;
  }

  // fill in ee
  auto const offset_laser = values.getValueMap().barrelItems().size();
  for (unsigned int i = 0; i < values.getValueMap().endcapItems().size(); i++) {
    p1_[offset_laser + i] = values.getValueMap().endcapItems()[i].p1;
    p2_[offset_laser + i] = values.getValueMap().endcapItems()[i].p2;
    p3_[offset_laser + i] = values.getValueMap().endcapItems()[i].p3;
  }

  //   Time is a simple std::vector
  //       typedef std::vector<EcalLaserTimeStamp> EcalLaserTimeStampMap;
  for (unsigned int i = 0; i < values.getTimeMap().size(); i++) {
    t1_[i] = values.getTimeMap()[i].t1.value();
    t2_[i] = values.getTimeMap()[i].t2.value();
    t3_[i] = values.getTimeMap()[i].t3.value();
  }
}

EcalLinearCorrectionsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(p1));
  cudaCheck(cudaFree(p2));
  cudaCheck(cudaFree(p3));
  cudaCheck(cudaFree(t1));
  cudaCheck(cudaFree(t2));
  cudaCheck(cudaFree(t3));
}

EcalLinearCorrectionsGPU::Product const& EcalLinearCorrectionsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalLinearCorrectionsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.p1, this->p1_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.p2, this->p2_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.p3, this->p3_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.t1, this->t1_.size() * sizeof(edm::TimeValue_t)));
        cudaCheck(cudaMalloc((void**)&product.t2, this->t2_.size() * sizeof(edm::TimeValue_t)));
        cudaCheck(cudaMalloc((void**)&product.t3, this->t3_.size() * sizeof(edm::TimeValue_t)));
        // transfer
        cudaCheck(cudaMemcpyAsync(
            product.p1, this->p1_.data(), this->p1_.size() * sizeof(float), cudaMemcpyHostToDevice, cudaStream));
        cudaCheck(cudaMemcpyAsync(
            product.p2, this->p2_.data(), this->p2_.size() * sizeof(float), cudaMemcpyHostToDevice, cudaStream));
        cudaCheck(cudaMemcpyAsync(
            product.p3, this->p3_.data(), this->p3_.size() * sizeof(float), cudaMemcpyHostToDevice, cudaStream));
        cudaCheck(cudaMemcpyAsync(product.t1,
                                  this->t1_.data(),
                                  this->t1_.size() * sizeof(edm::TimeValue_t),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.t2,
                                  this->t2_.data(),
                                  this->t2_.size() * sizeof(edm::TimeValue_t),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.t3,
                                  this->t3_.data(),
                                  this->t3_.size() * sizeof(edm::TimeValue_t),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalLinearCorrectionsGPU);
