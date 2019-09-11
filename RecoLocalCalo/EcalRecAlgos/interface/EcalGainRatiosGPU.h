#ifndef RecoLocalCalo_EcalRecProducers_src_EcalGainRatiosGPU_h
#define RecoLocalCalo_EcalRecProducers_src_EcalGainRatiosGPU_h

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAESProduct.h"
#endif

#include <cuda/api_wrappers.h>

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
  Product const& getProduct(cuda::stream_t<>&) const;

  //
  static std::string name() { return std::string{"ecalGainRatiosGPU"}; }

private:
  // in the future, we need to arrange so to avoid this copy on the host
  // store eb first then ee
  std::vector<float, CUDAHostAllocator<float>> gain12Over6_;
  std::vector<float, CUDAHostAllocator<float>> gain6Over1_;

  CUDAESProduct<Product> product_;

#endif
};

#endif
