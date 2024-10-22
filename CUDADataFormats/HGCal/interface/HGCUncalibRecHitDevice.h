#ifndef CUDADAtaFormats_HGCal_HGCUncalibRecHitDevice_H
#define CUDADAtaFormats_HGCal_HGCUncalibRecHitDevice_H

#include <cassert>
#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"

class HGCUncalibRecHitDevice {
public:
  HGCUncalibRecHitDevice() = default;
  explicit HGCUncalibRecHitDevice(uint32_t nhits, const cudaStream_t &stream);
  ~HGCUncalibRecHitDevice() = default;

  HGCUncalibRecHitDevice(const HGCUncalibRecHitDevice &) = delete;
  HGCUncalibRecHitDevice &operator=(const HGCUncalibRecHitDevice &) = delete;
  HGCUncalibRecHitDevice(HGCUncalibRecHitDevice &&) = default;
  HGCUncalibRecHitDevice &operator=(HGCUncalibRecHitDevice &&) = default;

  void defineSoAMemoryLayout_();
  HGCUncalibRecHitSoA get() const { return soa_; }
  uint32_t nHits() const { return nhits_; }
  uint32_t pad() const { return pad_; }
  uint32_t nBytes() const { return size_tot_; }

private:
  cms::cuda::device::unique_ptr<std::byte[]> ptr_;
  HGCUncalibRecHitSoA soa_;
  static constexpr std::array<int, memory::npointers::ntypes_hgcuncalibrechits_soa> sizes_ = {
      {memory::npointers::float_hgcuncalibrechits_soa * sizeof(float),
       memory::npointers::uint32_hgcuncalibrechits_soa * sizeof(uint32_t)}};

  uint32_t pad_;
  uint32_t nhits_;
  uint32_t size_tot_;
};

#endif  //CUDADAtaFormats_HGCal_HGCUncalibRecHitDevice_H
