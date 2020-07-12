#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalQIECodersGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalQIECodersGPU_h

#include "CondFormats/HcalObjects/interface/HcalQIEData.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalQIECodersGPU {
public:
  static constexpr uint32_t numValuesPerChannel = 16;

  struct Product {
    ~Product();
    float *offsets;
    float *slopes;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalQIECodersGPU(HcalQIEData const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalQIECodersGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;
  std::vector<float, cms::cuda::HostAllocator<float>> offsets_;
  std::vector<float, cms::cuda::HostAllocator<float>> slopes_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
