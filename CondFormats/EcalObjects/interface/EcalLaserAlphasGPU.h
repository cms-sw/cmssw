#ifndef CondFormats_EcalObjects_interface_EcalLaserAlphasGPU_h
#define CondFormats_EcalObjects_interface_EcalLaserAlphasGPU_h

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalLaserAlphasGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> values;
  };

#ifndef __CUDACC__
  //
  EcalLaserAlphasGPU(EcalLaserAlphas const&);

  // will call dealloation for Product thru ~Product
  ~EcalLaserAlphasGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  // TODO: do this centrally
  // get offset for hashes. equals number of barrel items
  uint32_t getOffset() const { return offset_; }

  //
  static std::string name() { return std::string{"ecalLaserAlphasGPU"}; }

private:
  std::vector<float, cms::cuda::HostAllocator<float>> values_;
  uint32_t offset_;

  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // CondFormats_EcalObjects_interface_EcalLaserAlphasGPU_h
