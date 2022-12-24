#ifndef RecoParticleFlow_PFClusterProducer_interface_HBHETopologyGPU_h
#define RecoParticleFlow_PFClusterProducer_interface_HBHETopologyGPU_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HBHETopologyGPU {
public:
  struct Product {
    ~Product();
    //int* values;
    //float3* pos; // KH CAUTION we need this one too later.
    uint32_t* detId;
    int* neighbours;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HBHETopologyGPU(edm::ParameterSet const&, const CaloGeometry &geom, const HcalTopology &topo);

  // will trigger deallocation of Product thru ~Product
  ~HBHETopologyGPU() = default;

  //std::vector<int, cms::cuda::HostAllocator<int>> const& getValues() const { return values_; }

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> const& getValuesDetId() const { return detId_; }
  std::vector<int, cms::cuda::HostAllocator<int>> const& getValuesNeighbours() const { return neighbours_; }
  
private:
  //std::vector<int, cms::cuda::HostAllocator<int>> values_;

  std::vector<uint, cms::cuda::HostAllocator<uint32_t>> detId_;
  std::vector<int, cms::cuda::HostAllocator<int>> neighbours_;
  
  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
