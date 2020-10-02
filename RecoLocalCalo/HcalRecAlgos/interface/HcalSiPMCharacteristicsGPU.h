#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalSiPMCharacteristicsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalSiPMCharacteristicsGPU_h

#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalSiPMCharacteristicsGPU {
public:
  struct Product {
    ~Product();
    int *pixels;
    float *parLin1, *parLin2, *parLin3;
    float *crossTalk;
    int *auxi1;
    float *auxi2;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalSiPMCharacteristicsGPU(HcalSiPMCharacteristics const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalSiPMCharacteristicsGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

private:
  std::vector<int, cms::cuda::HostAllocator<int>> pixels_, auxi1_;
  std::vector<float, cms::cuda::HostAllocator<float>> parLin1_, parLin2_, parLin3_, crossTalk_, auxi2_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
