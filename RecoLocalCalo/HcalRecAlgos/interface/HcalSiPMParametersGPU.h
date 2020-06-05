#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalSiPMParametersGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalSiPMParametersGPU_h

#include "CondFormats/HcalObjects/interface/HcalSiPMParameters.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalSiPMParametersGPU {
public:
  struct Product {
    ~Product();
    int *type, *auxi1;
    float *fcByPE, *darkCurrent, *auxi2;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalSiPMParametersGPU(HcalSiPMParameters const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalSiPMParametersGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;
  std::vector<int, cms::cuda::HostAllocator<int>> type_, auxi1_;
  std::vector<float, cms::cuda::HostAllocator<float>> fcByPE_, darkCurrent_, auxi2_;
  /*
  std::vector<float, cms::cuda::HostAllocator<float>> value0_, value1_, value2_, value3_, width0_, width1_, width2_, width3_;
  */

  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_HcalRecAlgos_interface_HcalSiPMParametersGPU_h
