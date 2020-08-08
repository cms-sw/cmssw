#ifndef RecoLocalCalo_EcalRecProducers_src_EcalRechitADCToGeVConstantGPU_h
#define RecoLocalCalo_EcalRecProducers_src_EcalRechitADCToGeVConstantGPU_h

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class EcalRechitADCToGeVConstantGPU {
public:
  struct Product {
    ~Product();
    float* adc2gev = nullptr;
  };

#ifndef __CUDACC__

  //
  EcalRechitADCToGeVConstantGPU(EcalADCToGeVConstant const&);

  // will call dealloation for Product thru ~Product
  ~EcalRechitADCToGeVConstantGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  //
  static std::string name() { return std::string{"ecalRechitADCToGeVConstantGPU"}; }

private:
  // in the future, we need to arrange so to avoid this copy on the host
  // store eb first then ee
  std::vector<float, cms::cuda::HostAllocator<float>> adc2gev_;

  cms::cuda::ESProduct<Product> product_;

#endif
};

#endif
