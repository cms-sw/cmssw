#ifndef RecoLocalCalo_EcalRecAlgos_interface_EcalPulseCovariancesGPU_h
#define RecoLocalCalo_EcalRecAlgos_interface_EcalPulseCovariancesGPU_h

#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalPulseCovariancesGPU {
public:
  struct Product {
    ~Product();
    EcalPulseCovariance* values = nullptr;
  };

#ifndef __CUDACC__
  // rearrange pedestals
  EcalPulseCovariancesGPU(EcalPulseCovariances const&);

  // will call dealloation for Product thru ~Product
  ~EcalPulseCovariancesGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  //
  static std::string name() { return std::string{"ecalPulseCovariancesGPU"}; }

private:
  // reuse original vectors (although with default allocator)
  std::vector<EcalPulseCovariance> const& valuesEB_;
  std::vector<EcalPulseCovariance> const& valuesEE_;

  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_EcalRecAlgos_interface_EcalPulseCovariancesGPU_h
