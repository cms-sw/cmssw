#ifndef RecoLocalCalo_EcalRecAlgos_interface_EcalMultifitParametersGPU_h
#define RecoLocalCalo_EcalRecAlgos_interface_EcalMultifitParametersGPU_h

#include <array>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalMultifitParametersGPU {
public:
  struct Product {
    ~Product();
    double *amplitudeFitParametersEB, *amplitudeFitParametersEE, *timeFitParametersEB, *timeFitParametersEE;
  };

#ifndef __CUDACC__
  EcalMultifitParametersGPU(edm::ParameterSet const&);

  ~EcalMultifitParametersGPU() = default;

  Product const& getProduct(cudaStream_t) const;

  std::array<std::reference_wrapper<std::vector<double, cms::cuda::HostAllocator<double>> const>, 4> getValues() const {
    return {{amplitudeFitParametersEB_, amplitudeFitParametersEE_, timeFitParametersEB_, timeFitParametersEE_}};
  }

private:
  std::vector<double, cms::cuda::HostAllocator<double>> amplitudeFitParametersEB_, amplitudeFitParametersEE_,
      timeFitParametersEB_, timeFitParametersEE_;

  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_EcalRecAlgos_interface_EcalMultifitParametersGPU_h
