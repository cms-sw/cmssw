#ifndef CondFormats_EcalObjects_interface_EcalRechitADCToGeVConstantGPU_h
#define CondFormats_EcalObjects_interface_EcalRechitADCToGeVConstantGPU_h

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalRechitADCToGeVConstantGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> adc2gev;
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

#endif  // __CUDACC__
};

#endif  // CondFormats_EcalObjects_interface_EcalRechitADCToGeVConstantGPU_h
