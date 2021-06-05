#ifndef CondFormats_EcalObjects_interface_EcalMultifitParametersGPU_h
#define CondFormats_EcalObjects_interface_EcalMultifitParametersGPU_h

#include <array>

#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalMultifitParametersGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> amplitudeFitParametersEB;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> amplitudeFitParametersEE;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> timeFitParametersEB;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> timeFitParametersEE;
  };

#ifndef __CUDACC__
  EcalMultifitParametersGPU(std::vector<double> const& amplitudeEB,
                            std::vector<double> const& amplitudeEE,
                            std::vector<double> const& timeEB,
                            std::vector<double> const& timeEE);

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

#endif  // CondFormats_EcalObjects_interface_EcalMultifitParametersGPU_h
