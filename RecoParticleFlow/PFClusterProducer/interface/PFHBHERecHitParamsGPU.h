#ifndef RecoParticleFlow_PFClusterProducer_interface_PFHBHERecHitParamsGPU_h
#define RecoParticleFlow_PFClusterProducer_interface_PFHBHERecHitParamsGPU_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class PFHBHERecHitParamsGPU {
public:
  struct Product {
    ~Product();
    /*
    int* depthHB;
    int* depthHE;
    float* thresholdE_HB;
    float* thresholdE_HE;
    */
    edm::propagate_const_array<cms::cuda::device::unique_ptr<int[]>> depthHB;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<int[]>> depthHE;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> thresholdE_HB;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> thresholdE_HE;
  };

#ifndef __CUDACC__
  // rearrange reco params
  PFHBHERecHitParamsGPU(edm::ParameterSet const&);

  // will trigger deallocation of Product thru ~Product
  ~PFHBHERecHitParamsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  //using intvec = std::reference_wrapper<std::vector<int, cms::cuda::HostAllocator<int>> const>;
  //using uint32vec = std::reference_wrapper<std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> const>;
  //using floatvec = std::reference_wrapper<std::vector<float, cms::cuda::HostAllocator<float>> const>;

  std::vector<int, cms::cuda::HostAllocator<int>> const& getValuesdepthHB() const { return depthHB_; }
  std::vector<int, cms::cuda::HostAllocator<int>> const& getValuesdepthHE() const { return depthHE_; }
  std::vector<float, cms::cuda::HostAllocator<float>> const& getValuesthresholdE_HB() const { return thresholdE_HB_; }
  std::vector<float, cms::cuda::HostAllocator<float>> const& getValuesthresholdE_HE() const { return thresholdE_HE_; }

private:

  std::vector<int, cms::cuda::HostAllocator<int>> depthHB_;
  std::vector<int, cms::cuda::HostAllocator<int>> depthHE_;
  std::vector<float, cms::cuda::HostAllocator<float>> thresholdE_HB_;
  std::vector<float, cms::cuda::HostAllocator<float>> thresholdE_HE_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
