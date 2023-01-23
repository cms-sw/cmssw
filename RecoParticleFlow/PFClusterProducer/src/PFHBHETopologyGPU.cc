#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHETopologyGPU.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigatorCore.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <unordered_set>

PFHBHETopologyGPU::PFHBHETopologyGPU(edm::ParameterSet const& ps,
				 const CaloGeometry& geom,
				 const HcalTopology& topo)
  : vhcalEnum_(ps.getParameter<std::vector<int>>("hcalEnums"))
{

  // Checking geom
  const CaloSubdetectorGeometry* hcalBarrelGeo = geom.getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const CaloSubdetectorGeometry* hcalEndcapGeo = geom.getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

  // Utilize PFHCALDenseIdNavigatorCore
  std::unique_ptr<PFHCALDenseIdNavigatorCore> navicore = std::make_unique<PFHCALDenseIdNavigatorCore>(vhcalEnum_,geom,topo);

  //
  // Filling HCAL DenseID vectors
  const std::vector<uint32_t> denseId = navicore.get()->getValidDenseIds();

  //
  // Filling information to define arrays for all relevant HBHE DetIds
  denseIdMax_ = *max_element(denseId.begin(), denseId.end());
  denseIdMin_ = *min_element(denseId.begin(), denseId.end());
  const int detIdArraySize = denseIdMax_ - denseIdMin_ + 1;

  //
  // Filling detId, positions, neighbours in arrays indexed based on denseId
  std::vector<uint32_t> detId;
  detId.clear();
  detId.resize(detIdArraySize);
  std::vector<float3> position;
  position.clear();
  position.resize(detIdArraySize);
  std::vector<int> neighbours;
  neighbours.clear();
  neighbours.resize(detIdArraySize*8);

  for (auto denseid : denseId) {

    DetId detid = topo.denseId2detId(denseid);
    HcalDetId hid = HcalDetId(detid);
    GlobalPoint pos;
    if (hid.subdet() == HcalBarrel) pos = hcalBarrelGeo->getGeometry(detid)->getPosition();
    else if (hid.subdet() == HcalEndcap) pos = hcalEndcapGeo->getGeometry(detid)->getPosition();
    else std::cout << "Unexpected subdetector found for detId " << hid.rawId() << ": " << hid.subdet() << std::endl;

    unsigned index = getIdx(denseid);
    detId[index] = (uint32_t)detid;
    position[index] = make_float3(pos.x(),pos.y(),pos.z());

    auto neigh = navicore.get()->getNeighbours(denseid);

    for (uint32_t n = 0; n < 8; n++) {
      // cmssdt.cern.ch/lxr/source/RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h#0087
      // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
      // neigh[0] is the rechit itself. Skip for neighbour array
      // If no neighbour exists in a direction, the value will be 0
      // Some neighbors from HF included! Need to test if these are included in the map!
      auto neighDetId = neigh[n + 1].rawId();
      if (neighDetId > 0
	  && (&topo)->detId2denseId(neighDetId)>=denseIdMin_
	  && (&topo)->detId2denseId(neighDetId)<=denseIdMax_) {
	neighbours[index * 8 + n] = getIdx(topo.detId2denseId(neighDetId));
      } else
	neighbours[index * 8 + n] = -1;
    }

  }

  //
  // Fill variables for HostAllocator
  denseId_.resize(denseId.size());
  std::copy(denseId.begin(), denseId.end(), denseId_.begin());
  //
  detId_.resize(detId.size());
  std::copy(detId.begin(), detId.end(), detId_.begin());
  neighbours_.resize(neighbours.size());
  std::copy(neighbours.begin(), neighbours.end(), neighbours_.begin());
  position_.resize(position.size());
  std::copy(position.begin(), position.end(), position_.begin());

  navicore.release();

}

PFHBHETopologyGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(denseId));
  cudaCheck(cudaFree(detId));
  cudaCheck(cudaFree(neighbours));
  cudaCheck(cudaFree(position));
}

PFHBHETopologyGPU::Product const& PFHBHETopologyGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](PFHBHETopologyGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        //cudaCheck(cudaMalloc((void**)&product.values, this->values_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.denseId, this->denseId_.size() * sizeof(uint32_t)));
        cudaCheck(cudaMalloc((void**)&product.detId, this->detId_.size() * sizeof(uint32_t)));
        cudaCheck(cudaMalloc((void**)&product.position, this->position_.size() * sizeof(float3)));
        cudaCheck(cudaMalloc((void**)&product.neighbours, this->neighbours_.size() * sizeof(int)));


        // transfer
        // cudaCheck(cudaMemcpyAsync(product.values,
        //                           this->values_.data(),
        //                           this->values_.size() * sizeof(int),
        //                           cudaMemcpyHostToDevice,
        //                           cudaStream));
        cudaCheck(cudaMemcpyAsync(product.denseId,
                                  this->denseId_.data(),
                                  this->denseId_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.detId,
                                  this->detId_.data(),
                                  this->detId_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.position,
                                  this->position_.data(),
                                  this->position_.size() * sizeof(float3),
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

TYPELOOKUP_DATA_REG(PFHBHETopologyGPU);
