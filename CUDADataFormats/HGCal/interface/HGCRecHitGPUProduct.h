#ifndef CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H
#define CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H

#include <cassert>
#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"
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
  void copySoAMemoryLayoutToConst_();
  HGCRecHitSoA get() { return soa_; }
  ConstHGCRecHitSoA get() const {
    ConstHGCRecHitSoA soa;
    soa.energy_    = reinterpret_cast<float const*>(mem_.get());
    soa.time_      = soa.energy_ + pad_;
    soa.timeError_ = soa.time_ + pad_;
    soa.id_        = reinterpret_cast<uint32_t const*>(soa.timeError_ + pad_);
    soa.flagBits_  = soa.id_ + pad_;
    soa.son_       = reinterpret_cast<uint8_t const*>(soa.flagBits_ + pad_);
    return soa;
  }

  //number of hits stored in the SoA
  uint32_t nHits() const { return nhits_; }
  //pad of memory block (used for warp alignment, slighlty larger than 'nhits_')
  uint32_t pad() const { return pad_; }
  //number of bytes of the SoA
  uint32_t nBytes() const { return size_tot_; }

private:
  cms::cuda::device::unique_ptr<std::byte[]> mem_;
  HGCRecHitSoA soa_;
  uint32_t pad_;
  uint32_t nhits_;
  uint32_t size_tot_;
};

#endif  //CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H
