#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalGainsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalGainsGPU_h

#include "CondFormats/HcalObjects/interface/HcalGains.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalGainsGPU {
public:
  struct Product {
    ~Product();
    float* values;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalGainsGPU(HcalGains const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalGainsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;
  std::vector<float, cms::cuda::HostAllocator<float>> values_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
