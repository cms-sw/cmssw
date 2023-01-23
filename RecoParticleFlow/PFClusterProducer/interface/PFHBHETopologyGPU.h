#ifndef RecoParticleFlow_PFClusterProducer_interface_PFHBHETopologyGPU_h
#define RecoParticleFlow_PFClusterProducer_interface_PFHBHETopologyGPU_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class PFHBHETopologyGPU {
public:
  struct Product {
    ~Product();
    //int* values;
    //uint32_t size;
    //uint32_t nChannels;
    //uint32_t topoArraySize;
    uint32_t* denseId;
    uint32_t* detId;
    float3* position;
    int* neighbours; // size of array is x8
  };

#ifndef __CUDACC__
  // rearrange reco params
  PFHBHETopologyGPU(edm::ParameterSet const&, const CaloGeometry &geom, const HcalTopology &topo);

  // will trigger deallocation of Product thru ~Product
  ~PFHBHETopologyGPU() = default;

  //std::vector<int, cms::cuda::HostAllocator<int>> const& getValues() const { return values_; }

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> const& getValuesDenseId() const { return denseId_; }
  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> const& getValuesDetId() const { return detId_; }
  std::vector<int, cms::cuda::HostAllocator<int>> const& getValuesNeighbours() const { return neighbours_; }
  std::vector<float3, cms::cuda::HostAllocator<float3>> const& getValuesPosition() const { return position_; }

private:
  //std::vector<int, cms::cuda::HostAllocator<int>> values_;

  std::vector<uint, cms::cuda::HostAllocator<uint32_t>> denseId_;
  std::vector<uint, cms::cuda::HostAllocator<uint32_t>> detId_;
  std::vector<int, cms::cuda::HostAllocator<int>> neighbours_;
  std::vector<float3, cms::cuda::HostAllocator<float3>> position_;

  cms::cuda::ESProduct<Product> product_;

  //for internal use
  unsigned int denseIdMax_;
  unsigned int denseIdMin_;
  std::vector<int> vhcalEnum_;

  unsigned int getIdx(const unsigned int denseid) const {
    unsigned index = denseid - denseIdMin_;
    return index;
  }

#endif
};

#endif
