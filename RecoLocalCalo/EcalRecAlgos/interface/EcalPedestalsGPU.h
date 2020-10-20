#ifndef RecoLocalCalo_EcalRecAlgos_interface_EcalPedestalsGPU_h
#define RecoLocalCalo_EcalRecAlgos_interface_EcalPedestalsGPU_h

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalPedestalsGPU {
public:
  struct Product {
    ~Product();
    float *mean_x12 = nullptr, *mean_x6 = nullptr, *mean_x1 = nullptr;
    float *rms_x12 = nullptr, *rms_x6 = nullptr, *rms_x1 = nullptr;
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

#endif  // RecoLocalCalo_EcalRecAlgos_interface_EcalPedestalsGPU_h
