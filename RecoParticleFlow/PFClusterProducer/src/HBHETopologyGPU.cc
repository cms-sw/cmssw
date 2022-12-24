#include "RecoParticleFlow/PFClusterProducer/interface/HBHETopologyGPU.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

HBHETopologyGPU::HBHETopologyGPU(edm::ParameterSet const& ps,
				 const CaloGeometry& geom,
				 const HcalTopology& topo){
				 // const edm::ESHandle<CaloGeometry>& geoHandle,
				 // const edm::ESHandle<HcalTopology>& topoHandle) {

  const CaloSubdetectorGeometry* hcalBarrelGeo = geom.getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const CaloSubdetectorGeometry* hcalEndcapGeo = geom.getSubdetectorGeometry(DetId::Hcal, HcalEndcap);
  
  const std::vector<DetId>& validBarrelDetIds = hcalBarrelGeo->getValidDetIds(DetId::Hcal, HcalBarrel);
  const std::vector<DetId>& validEndcapDetIds = hcalEndcapGeo->getValidDetIds(DetId::Hcal, HcalEndcap);
  
  std::cout << "HBHETopologyGPU test " << validBarrelDetIds.size() << " " << validEndcapDetIds.size() << std::endl;
  std::cout << "HBHETopologyGPU constructor" << std::endl;

  // KH - of course these are dummy 
  auto const& detId = ps.getParameter<std::vector<uint32_t>>("pulseOffsets");
  auto const& neighbours = ps.getParameter<std::vector<int>>("pulseOffsets2");
  detId_.resize(detId.size());
  std::copy(detId.begin(), detId.end(), detId_.begin());
  neighbours_.resize(neighbours.size());
  std::copy(neighbours.begin(), neighbours.end(), neighbours_.begin());
}

HBHETopologyGPU::Product::~Product() {
  // deallocation
  //cudaCheck(cudaFree(pos));
  cudaCheck(cudaFree(detId));
  cudaCheck(cudaFree(neighbours));
}

HBHETopologyGPU::Product const& HBHETopologyGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HBHETopologyGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        //cudaCheck(cudaMalloc((void**)&product.values, this->values_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.detId, this->detId_.size() * sizeof(uint32_t)));
        cudaCheck(cudaMalloc((void**)&product.neighbours, this->neighbours_.size() * sizeof(int)));

        // transfer
        // cudaCheck(cudaMemcpyAsync(product.values,
        //                           this->values_.data(),
        //                           this->values_.size() * sizeof(int),
        //                           cudaMemcpyHostToDevice,
        //                           cudaStream));
        cudaCheck(cudaMemcpyAsync(product.detId,
                                  this->detId_.data(),
                                  this->detId_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.neighbours,
                                  this->neighbours_.data(),
                                  this->neighbours_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HBHETopologyGPU);
