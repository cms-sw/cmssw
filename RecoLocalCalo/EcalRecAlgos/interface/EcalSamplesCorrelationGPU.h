#ifndef RecoLocalCalo_EcalRecAlgos_interface_EcalSamplesCorrelationGPU_h
#define RecoLocalCalo_EcalRecAlgos_interface_EcalSamplesCorrelationGPU_h

#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalSamplesCorrelationGPU {
public:
  struct Product {
    ~Product();
    double *EBG12SamplesCorrelation = nullptr, *EBG6SamplesCorrelation = nullptr, *EBG1SamplesCorrelation = nullptr;
    double *EEG12SamplesCorrelation = nullptr, *EEG6SamplesCorrelation = nullptr, *EEG1SamplesCorrelation = nullptr;
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
  std::vector<double> const& EBG12SamplesCorrelation_;
  std::vector<double> const& EBG6SamplesCorrelation_;
  std::vector<double> const& EBG1SamplesCorrelation_;
  std::vector<double> const& EEG12SamplesCorrelation_;
  std::vector<double> const& EEG6SamplesCorrelation_;
  std::vector<double> const& EEG1SamplesCorrelation_;

  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_EcalRecAlgos_interface_EcalSamplesCorrelationGPU_h
