#ifndef RecoParticleFlow_PFClusterProducer_interface_HBHETopologyGPU_h
#define RecoParticleFlow_PFClusterProducer_interface_HBHETopologyGPU_h

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

  //for internal use
  std::vector<unsigned int> vDenseIdHcal_;
  std::vector<std::vector<DetId>> neighboursHcal_;
  unsigned int denseIdHcalMax_;
  unsigned int denseIdHcalMin_;

  bool validNeighbours(const unsigned int denseid) const {
    bool ok = true;
    unsigned index = getIdx(denseid);
    if (index >= neighboursHcal_.size() || neighboursHcal_.at(index).size() != 9)
      ok = false;  // the neighbour vector size should be 3x3
    return ok;
  }

  unsigned int getIdx(const unsigned int denseid) const {
    unsigned index = denseid - denseIdHcalMin_;
    return index;
  }

#endif
};

#endif
