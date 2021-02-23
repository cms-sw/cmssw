#ifndef CUDADAtaFormats_HGCal_HGCRecHitCPUProduct_H
#define CUDADAtaFormats_HGCal_HGCRecHitCPUProduct_H

#include <cassert>
#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"

class HGCRecHitCPUProduct {
public:
  HGCRecHitCPUProduct() = default;
  explicit HGCRecHitCPUProduct(uint32_t nhits, const cudaStream_t &stream);
  ~HGCRecHitCPUProduct() = default;

  HGCRecHitCPUProduct(const HGCRecHitCPUProduct &) = delete;
  HGCRecHitCPUProduct &operator=(const HGCRecHitCPUProduct &) = delete;
  HGCRecHitCPUProduct(HGCRecHitCPUProduct &&) = default;
  HGCRecHitCPUProduct &operator=(HGCRecHitCPUProduct &&) = default;

  void defineSoAMemoryLayout_();
  void copySoAMemoryLayoutToConst_();
  HGCRecHitSoA get() { return soa_; }
  //copy at request time prevents users from modifying the original pointers in `constSoa`
  ConstHGCRecHitSoA get() const { return constSoa_; }
  uint32_t nHits() const { return nhits_; }
  uint32_t pad() const { return pad_; }
  uint32_t nBytes() const { return size_tot_; }

private:
  //cms::cuda::host::unique_ptr<std::byte[]> mem_;
  std::unique_ptr<std::byte[]> mem_;
  HGCRecHitSoA soa_;
  ConstHGCRecHitSoA constSoa_;
  uint32_t pad_;
  uint32_t nhits_;
  uint32_t size_tot_;
};

#endif  //CUDADAtaFormats_HGCal_HGCRecHitCPUProduct_H
