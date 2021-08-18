#ifndef CondFormats_EcalObjects_interface_EcalLinearCorrectionsGPU_h
#define CondFormats_EcalObjects_interface_EcalLinearCorrectionsGPU_h

#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalLinearCorrectionsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> p1;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> p2;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> p3;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<edm::TimeValue_t[]>> t1;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<edm::TimeValue_t[]>> t2;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<edm::TimeValue_t[]>> t3;
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

#endif  // CondFormats_EcalObjects_interface_EcalLinearCorrectionsGPU_h
