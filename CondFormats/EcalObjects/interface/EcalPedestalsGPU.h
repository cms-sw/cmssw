#ifndef CondFormats_EcalObjects_interface_EcalPedestalsGPU_h
#define CondFormats_EcalObjects_interface_EcalPedestalsGPU_h

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalPedestalsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> mean_x12;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> mean_x6;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> mean_x1;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> rms_x12;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> rms_x6;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> rms_x1;
  };

#ifndef __CUDACC__

  // rearrange pedestals
  EcalPedestalsGPU(EcalPedestals const &);

  // will call dealloation for Product thru ~Product
  ~EcalPedestalsGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

  //
  static std::string name() { return std::string{"ecalPedestalsGPU"}; }

private:
  // in the future, we need to arrange so to avoid this copy on the host
  // store eb first then ee
  std::vector<float, cms::cuda::HostAllocator<float>> mean_x12_;
  std::vector<float, cms::cuda::HostAllocator<float>> rms_x12_;
  std::vector<float, cms::cuda::HostAllocator<float>> mean_x6_;
  std::vector<float, cms::cuda::HostAllocator<float>> rms_x6_;
  std::vector<float, cms::cuda::HostAllocator<float>> mean_x1_;
  std::vector<float, cms::cuda::HostAllocator<float>> rms_x1_;

  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // CondFormats_EcalObjects_interface_EcalPedestalsGPU_h
