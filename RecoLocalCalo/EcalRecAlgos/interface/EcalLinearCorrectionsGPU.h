#ifndef RecoLocalCalo_EcalRecAlgos_interface_EcalLinearCorrectionsGPU_h
#define RecoLocalCalo_EcalRecAlgos_interface_EcalLinearCorrectionsGPU_h

#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalLinearCorrectionsGPU {
public:
  struct Product {
    ~Product();
    float *p1 = nullptr;
    float *p2 = nullptr;
    float *p3 = nullptr;
    edm::TimeValue_t *t1 = nullptr;
    edm::TimeValue_t *t2 = nullptr;
    edm::TimeValue_t *t3 = nullptr;
  };

#ifndef __CUDACC__

  //
  EcalLinearCorrectionsGPU(EcalLinearCorrections const &);

  // will call dealloation for Product thru ~Product
  ~EcalLinearCorrectionsGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

  //
  static std::string name() { return std::string{"ecalLinearCorrectionsGPU"}; }

private:
  // in the future, we need to arrange so to avoid this copy on the host
  // store eb first then ee
  std::vector<float, cms::cuda::HostAllocator<float>> p1_;
  std::vector<float, cms::cuda::HostAllocator<float>> p2_;
  std::vector<float, cms::cuda::HostAllocator<float>> p3_;

  std::vector<edm::TimeValue_t, cms::cuda::HostAllocator<edm::TimeValue_t>> t1_;
  std::vector<edm::TimeValue_t, cms::cuda::HostAllocator<edm::TimeValue_t>> t2_;
  std::vector<edm::TimeValue_t, cms::cuda::HostAllocator<edm::TimeValue_t>> t3_;

  cms::cuda::ESProduct<Product> product_;

#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_EcalRecAlgos_interface_EcalLinearCorrectionsGPU_h
