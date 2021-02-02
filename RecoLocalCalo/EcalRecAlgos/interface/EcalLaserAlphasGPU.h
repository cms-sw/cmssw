#ifndef RecoLocalCalo_EcalRecAlgos_interface_EcalLaserAlphasGPU_h
#define RecoLocalCalo_EcalRecAlgos_interface_EcalLaserAlphasGPU_h

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalLaserAlphasGPU {
public:
  struct Product {
    ~Product();
    float* values = nullptr;
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
  uint32_t getOffset() const { return valuesEB_.size(); }

  //
  static std::string name() { return std::string{"ecalLaserAlphasGPU"}; }

private:
  std::vector<float> const& valuesEB_;
  std::vector<float> const& valuesEE_;

  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_EcalRecAlgos_interface_EcalLaserAlphasGPU_h
