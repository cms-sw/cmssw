#ifndef RecoParticleFlow_PFClusterProducer_interface_PFHBHERecHitParamsGPU_h
#define RecoParticleFlow_PFClusterProducer_interface_PFHBHERecHitParamsGPU_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class PFHBHERecHitParamsGPU {
public:
  struct Product {
    ~Product();
    int* valuesdepthHB;
    int* valuesdepthHE;
    double* valuesthresholdE_HB;
    double* valuesthresholdE_HE;
    /*
    edm::propagate_const_array<cms::cuda::device::unique_ptr<uint32_t[]>> depthHB;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<uint32_t[]>> depthHE;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> thresholdE_HB;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<double[]>> thresholdE_HE;
    */
  };

#ifndef __CUDACC__
  // rearrange reco params
  PFHBHERecHitParamsGPU(edm::ParameterSet const&);
  /*
  PFHBHERecHitParamsGPU(std::vector<unit32_t> const& depthHB,
			std::vector<unit32_t> const& depthHE,
			std::vector<double> const& thresholdE_HB,
			std::vector<double> const& thresholdE_HE
			);
  */

  // will trigger deallocation of Product thru ~Product
  ~PFHBHERecHitParamsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  using intvec = std::reference_wrapper<std::vector<int, cms::cuda::HostAllocator<int>> const>;
  using uint32vec = std::reference_wrapper<std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> const>;
  using doublevec = std::reference_wrapper<std::vector<double, cms::cuda::HostAllocator<double>> const>;
  
  std::vector<int, cms::cuda::HostAllocator<int>> const& getValuesdepthHB() const { return valuesdepthHB_; }
  std::vector<int, cms::cuda::HostAllocator<int>> const& getValuesdepthHE() const { return valuesdepthHE_; }
  std::vector<double, cms::cuda::HostAllocator<double>> const& getValuesthresholdE_HB() const { return valuesthresholdE_HB_; }
  std::vector<double, cms::cuda::HostAllocator<double>> const& getValuesthresholdE_HE() const { return valuesthresholdE_HE_; }

  /* std::array<std::reference_wrapper<std::vector<double, cms::cuda::HostAllocator<double>> const>, 4> getValues() const { */
  /*   return {{depthHB_, depthHE_, thresholdE_HB_, thresholdE_HE_}}; */
  /* } */

  /*
  std::tuple<uint32vec, doublevec> getValuesHB() const {
    return {valuesdepthHB_, valuesthresholdE_HB_};
  }
  */

private:
  std::vector<int, cms::cuda::HostAllocator<int>> valuesdepthHB_;
  std::vector<int, cms::cuda::HostAllocator<int>> valuesdepthHE_;
  std::vector<double, cms::cuda::HostAllocator<double>> valuesthresholdE_HB_;
  std::vector<double, cms::cuda::HostAllocator<double>> valuesthresholdE_HE_;
  /*
  std::vector<int, cms::cuda::HostAllocator<int>> depthHB_;
  std::vector<int, cms::cuda::HostAllocator<int>> depthHE_;
  std::vector<double, cms::cuda::HostAllocator<double>> thresholdE_HB_;
  std::vector<double, cms::cuda::HostAllocator<double>> thresholdE_HE_;
  */

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
