#ifndef CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H
#define CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H

#include <cassert>
#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibratedRecHitSoA.h"

class HGCRecHitGPUProduct {
public:
  HGCRecHitGPUProduct() = default;
  explicit HGCRecHitGPUProduct(const uint32_t &nhits, const cudaStream_t &stream);
  ~HGCRecHitGPUProduct() = default;

  HGCRecHitGPUProduct(const HGCRecHitGPUProduct &) = delete;
  HGCRecHitGPUProduct &operator=(const HGCRecHitGPUProduct &) = delete;
  HGCRecHitGPUProduct(HGCRecHitGPUProduct &&) = default;
  HGCRecHitGPUProduct &operator=(HGCRecHitGPUProduct &&) = default;

  std::byte *get() const { return ptr_.get(); }

  uint32_t nHits() const { return nhits_; }
  uint32_t stride() const { return stride_; }
  uint32_t nBytes() const { return size_tot_; }

private:
  cms::cuda::device::unique_ptr<std::byte[]> ptr_;
  uint32_t stride_;
  uint32_t nhits_;
  uint32_t size_tot_;
};

#endif  //CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H
