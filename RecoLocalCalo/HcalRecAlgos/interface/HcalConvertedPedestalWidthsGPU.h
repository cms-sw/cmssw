#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalConvertedPedestalWidthsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalConvertedPedestalWidthsGPU_h

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalQIETypes.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalConvertedPedestalWidthsGPU {
public:
  struct Product {
    ~Product();
    float* values;
  };

#ifndef __CUDACC__
  // order matters!
  HcalConvertedPedestalWidthsGPU(HcalPedestals const&,
                                 HcalPedestalWidths const&,
                                 HcalQIEData const&,
                                 HcalQIETypes const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalConvertedPedestalWidthsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;
  std::vector<float, cms::cuda::HostAllocator<float>> values_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
