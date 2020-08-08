#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalTimeCorrsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalTimeCorrsGPU_h

#include "CondFormats/HcalObjects/interface/HcalTimeCorrs.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalTimeCorrsGPU {
public:
  struct Product {
    ~Product();
    float* value;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalTimeCorrsGPU(HcalTimeCorrs const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalTimeCorrsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

private:
  std::vector<float, cms::cuda::HostAllocator<float>> value_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
