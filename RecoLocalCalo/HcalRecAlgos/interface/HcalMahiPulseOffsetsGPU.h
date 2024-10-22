#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalMahiPulseOffsetsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalMahiPulseOffsetsGPU_h

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalMahiPulseOffsetsGPU {
public:
  struct Product {
    ~Product();
    int* values;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalMahiPulseOffsetsGPU(std::vector<int> const& values);

  // will trigger deallocation of Product thru ~Product
  ~HcalMahiPulseOffsetsGPU() = default;

  std::vector<int, cms::cuda::HostAllocator<int>> const& getValues() const { return values_; }

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

private:
  std::vector<int, cms::cuda::HostAllocator<int>> values_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
