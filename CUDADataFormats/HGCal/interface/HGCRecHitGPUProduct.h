#ifndef CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H
#define CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H

#include <cassert>
#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"

class HGCRecHitGPUProduct {
public:
  HGCRecHitGPUProduct() = default;
  explicit HGCRecHitGPUProduct(uint32_t nhits, const cudaStream_t &stream);
  ~HGCRecHitGPUProduct() = default;

  HGCRecHitGPUProduct(const HGCRecHitGPUProduct &) = delete;
  HGCRecHitGPUProduct &operator=(const HGCRecHitGPUProduct &) = delete;
  HGCRecHitGPUProduct(HGCRecHitGPUProduct &&) = default;
  HGCRecHitGPUProduct &operator=(HGCRecHitGPUProduct &&) = default;

  void defineSoAMemoryLayout_();
  HGCRecHitSoA get() const { return soa_; }
  uint32_t nHits() const { return nhits_; }
  uint32_t pad() const { return pad_; }
  uint32_t nBytes() const { return size_tot_; }

private:
  cms::cuda::device::unique_ptr<std::byte[]> ptr_;
  HGCRecHitSoA soa_;
  uint32_t pad_;
  uint32_t nhits_;
  uint32_t size_tot_;
};

#endif  //CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H
