#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalGainWidthsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalGainWidthsGPU_h

#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalGainWidthsGPU {
public:
  struct Product {
    ~Product();
    float *value0;
    float *value1;
    float *value2;
    float *value3;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalGainWidthsGPU(HcalGainWidths const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalGainWidthsGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;
  std::vector<float, cms::cuda::HostAllocator<float>> value0_, value1_, value2_, value3_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
