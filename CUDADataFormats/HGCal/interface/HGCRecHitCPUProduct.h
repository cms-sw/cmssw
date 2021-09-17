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
  explicit HGCRecHitCPUProduct(uint32_t nhits, const cudaStream_t &stream) : nhits_(nhits) {
    size_tot_ = std::accumulate(sizes_.begin(), sizes_.end(), 0);  //this might be done at compile time
    pad_ = ((nhits - 1) / 32 + 1) * 32;                            //align to warp boundary (assumption: warpSize = 32)
    mem_ = cms::cuda::make_host_unique<std::byte[]>(pad_ * size_tot_, stream);
  }
  ~HGCRecHitCPUProduct() = default;

  HGCRecHitCPUProduct(const HGCRecHitCPUProduct &) = delete;
  HGCRecHitCPUProduct &operator=(const HGCRecHitCPUProduct &) = delete;
  HGCRecHitCPUProduct(HGCRecHitCPUProduct &&) = default;
  HGCRecHitCPUProduct &operator=(HGCRecHitCPUProduct &&) = default;

  HGCRecHitSoA get() {
    HGCRecHitSoA soa;
    soa.energy_ = reinterpret_cast<float *>(mem_.get());
    soa.time_ = soa.energy_ + pad_;
    soa.timeError_ = soa.time_ + pad_;
    soa.id_ = reinterpret_cast<uint32_t *>(soa.timeError_ + pad_);
    soa.flagBits_ = soa.id_ + pad_;
    soa.son_ = reinterpret_cast<uint8_t *>(soa.flagBits_ + pad_);
    soa.nbytes_ = size_tot_;
    soa.nhits_ = nhits_;
    soa.pad_ = pad_;
    return soa;
  }
  ConstHGCRecHitSoA get() const {
    ConstHGCRecHitSoA soa;
    soa.energy_ = reinterpret_cast<float const *>(mem_.get());
    soa.time_ = soa.energy_ + pad_;
    soa.timeError_ = soa.time_ + pad_;
    soa.id_ = reinterpret_cast<uint32_t const *>(soa.timeError_ + pad_);
    soa.flagBits_ = soa.id_ + pad_;
    soa.son_ = reinterpret_cast<uint8_t const *>(soa.flagBits_ + pad_);
    return soa;
  }
  uint32_t nHits() const { return nhits_; }
  uint32_t pad() const { return pad_; }
  uint32_t nBytes() const { return size_tot_; }

private:
  cms::cuda::host::unique_ptr<std::byte[]> mem_;
  static constexpr std::array<int, memory::npointers::ntypes_hgcrechits_soa> sizes_ = {
      {memory::npointers::float_hgcrechits_soa * sizeof(float),
       memory::npointers::uint32_hgcrechits_soa * sizeof(uint32_t),
       memory::npointers::uint8_hgcrechits_soa * sizeof(uint8_t)}};
  uint32_t pad_;
  uint32_t nhits_;
  uint32_t size_tot_;
};

#endif  //CUDADAtaFormats_HGCal_HGCRecHitCPUProduct_H
