#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitDevice.h"

HGCUncalibRecHitDevice::HGCUncalibRecHitDevice(uint32_t nhits, const cudaStream_t& stream) : nhits_(nhits) {
  constexpr std::array<int, memory::npointers::ntypes_hgcuncalibrechits_soa> sizes = {
      {memory::npointers::float_hgcuncalibrechits_soa * sizeof(float),
       memory::npointers::uint32_hgcuncalibrechits_soa * sizeof(uint32_t)}};

  size_tot_ = std::accumulate(sizes.begin(), sizes.end(), 0);
  pad_ = ((nhits - 1) / 32 + 1) * 32;  //align to warp boundary (assumption: warpSize = 32)
  ptr_ = cms::cuda::make_device_unique<std::byte[]>(pad_ * size_tot_, stream);

  defineSoAMemoryLayout_();
}

void HGCUncalibRecHitDevice::defineSoAMemoryLayout_() {
  soa_.amplitude_ = reinterpret_cast<float*>(ptr_.get());
  soa_.pedestal_ = soa_.amplitude_ + pad_;
  soa_.jitter_ = soa_.pedestal_ + pad_;
  soa_.chi2_ = soa_.jitter_ + pad_;
  soa_.OOTamplitude_ = soa_.chi2_ + pad_;
  soa_.OOTchi2_ = soa_.OOTamplitude_ + pad_;
  soa_.flags_ = reinterpret_cast<uint32_t*>(soa_.OOTchi2_ + pad_);
  soa_.aux_ = soa_.flags_ + pad_;
  soa_.id_ = soa_.aux_ + pad_;

  soa_.nbytes_ = size_tot_;
  soa_.nhits_ = nhits_;
  soa_.pad_ = pad_;
}
