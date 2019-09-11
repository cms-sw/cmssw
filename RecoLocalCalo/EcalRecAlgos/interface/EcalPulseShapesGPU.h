#ifndef RecoLocalCalo_EcalRecProducers_src_EcalPulseShapesGPU_h
#define RecoLocalCalo_EcalRecProducers_src_EcalPulseShapesGPU_h

#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAESProduct.h"
#endif

#include <cuda/api_wrappers.h>

class EcalPulseShapesGPU {
public:
  struct Product {
    ~Product();
    EcalPulseShape* values = nullptr;
  };

#ifndef __CUDACC__
  // rearrange pedestals
  EcalPulseShapesGPU(EcalPulseShapes const&);

  // will call dealloation for Product thru ~Product
  ~EcalPulseShapesGPU() = default;

  // get device pointers
  Product const& getProduct(cuda::stream_t<>&) const;

  //
  static std::string name() { return std::string{"ecalPulseShapesGPU"}; }

private:
  // reuse original vectors (although with default allocator)
  std::vector<EcalPulseShape> const& valuesEB_;
  std::vector<EcalPulseShape> const& valuesEE_;

  CUDAESProduct<Product> product_;
#endif
};

#endif
