#ifndef RecoLocalCalo_EcalRecAlgos_interface_EcalPulseShapesGPU_h
#define RecoLocalCalo_EcalRecAlgos_interface_EcalPulseShapesGPU_h

#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

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
  Product const& getProduct(cudaStream_t) const;

  //
  static std::string name() { return std::string{"ecalPulseShapesGPU"}; }

private:
  // reuse original vectors (although with default allocator)
  std::vector<EcalPulseShape> const& valuesEB_;
  std::vector<EcalPulseShape> const& valuesEE_;

  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_EcalRecAlgos_interface_EcalPulseShapesGPU_h
