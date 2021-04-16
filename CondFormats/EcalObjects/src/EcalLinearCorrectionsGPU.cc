#include "CondFormats/EcalObjects/interface/EcalLinearCorrectionsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

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

EcalLinearCorrectionsGPU::Product const& EcalLinearCorrectionsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalLinearCorrectionsGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.p1 = cms::cuda::make_device_unique<float[]>(p1_.size(), cudaStream);
        product.p2 = cms::cuda::make_device_unique<float[]>(p2_.size(), cudaStream);
        product.p3 = cms::cuda::make_device_unique<float[]>(p3_.size(), cudaStream);
        product.t1 = cms::cuda::make_device_unique<edm::TimeValue_t[]>(t1_.size(), cudaStream);
        product.t2 = cms::cuda::make_device_unique<edm::TimeValue_t[]>(t2_.size(), cudaStream);
        product.t3 = cms::cuda::make_device_unique<edm::TimeValue_t[]>(t3_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.p1, p1_, cudaStream);
        cms::cuda::copyAsync(product.p2, p2_, cudaStream);
        cms::cuda::copyAsync(product.p3, p3_, cudaStream);
        cms::cuda::copyAsync(product.t1, t1_, cudaStream);
        cms::cuda::copyAsync(product.t2, t2_, cudaStream);
        cms::cuda::copyAsync(product.t3, t3_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalLinearCorrectionsGPU);
