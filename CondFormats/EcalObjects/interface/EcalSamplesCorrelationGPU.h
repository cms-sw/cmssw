#ifndef CondFormats_EcalObjects_interface_EcalSamplesCorrelationGPU_h
#define CondFormats_EcalObjects_interface_EcalSamplesCorrelationGPU_h

#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalSamplesCorrelationGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> EBG12SamplesCorrelation;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> EBG6SamplesCorrelation;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> EBG1SamplesCorrelation;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> EEG12SamplesCorrelation;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> EEG6SamplesCorrelation;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> EEG1SamplesCorrelation;
  };

#ifndef __CUDACC__
  // rearrange pedestals
  EcalSamplesCorrelationGPU(EcalSamplesCorrelation const&);

  // will call dealloation for Product thru ~Product
  ~EcalSamplesCorrelationGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  //
  static std::string name() { return std::string{"ecalSamplesCorrelationGPU"}; }

private:
  std::vector<double, cms::cuda::HostAllocator<double>> EBG12SamplesCorrelation_;
  std::vector<double, cms::cuda::HostAllocator<double>> EBG6SamplesCorrelation_;
  std::vector<double, cms::cuda::HostAllocator<double>> EBG1SamplesCorrelation_;
  std::vector<double, cms::cuda::HostAllocator<double>> EEG12SamplesCorrelation_;
  std::vector<double, cms::cuda::HostAllocator<double>> EEG6SamplesCorrelation_;
  std::vector<double, cms::cuda::HostAllocator<double>> EEG1SamplesCorrelation_;

  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // CondFormats_EcalObjects_interface_EcalSamplesCorrelationGPU_h
