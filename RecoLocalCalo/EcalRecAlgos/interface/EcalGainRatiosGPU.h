#ifndef RecoLocalCalo_EcalRecAlgos_interface_EcalGainRatiosGPU_h
#define RecoLocalCalo_EcalRecAlgos_interface_EcalGainRatiosGPU_h

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalGainRatiosGPU {
public:
  struct Product {
    ~Product();
    float *gain12Over6 = nullptr, *gain6Over1 = nullptr;
  };

#ifndef __CUDACC__

  // rearrange pedestals
  EcalGainRatiosGPU(EcalGainRatios const&);

  // will call dealloation for Product thru ~Product
  ~EcalGainRatiosGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  //
  static std::string name() { return std::string{"ecalGainRatiosGPU"}; }

private:
  // in the future, we need to arrange so to avoid this copy on the host
  // store eb first then ee
  std::vector<float, cms::cuda::HostAllocator<float>> gain12Over6_;
  std::vector<float, cms::cuda::HostAllocator<float>> gain6Over1_;

  cms::cuda::ESProduct<Product> product_;

#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_EcalRecAlgos_interface_EcalGainRatiosGPU_h
