#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalPedestalWidthsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalPedestalWidthsGPU_h

#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalPedestalWidthsGPU {
public:
  struct Product {
    ~Product();
    float *sigma00, *sigma01, *sigma02, *sigma03, *sigma10, *sigma11, *sigma12, *sigma13, *sigma20, *sigma21, *sigma22,
        *sigma23, *sigma30, *sigma31, *sigma32, *sigma33;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalPedestalWidthsGPU(HcalPedestalWidths const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalPedestalWidthsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  // as in cpu version
  bool unitIsADC() const { return unitIsADC_; }

private:
  bool unitIsADC_;
  uint64_t totalChannels_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma00_, sigma01_, sigma02_, sigma03_, sigma10_, sigma11_,
      sigma12_, sigma13_, sigma20_, sigma21_, sigma22_, sigma23_, sigma30_, sigma31_, sigma32_, sigma33_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
