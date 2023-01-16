#include "RecoParticleFlow/PFClusterProducer/interface/HBHETopologyGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <unordered_set>

HBHETopologyGPU::HBHETopologyGPU(edm::ParameterSet const& ps,
				 const CaloGeometry& geom,
				 const HcalTopology& topo){
				 // const edm::ESHandle<CaloGeometry>& geoHandle,
				 // const edm::ESHandle<HcalTopology>& topoHandle) {

  // Checking geom
  const CaloSubdetectorGeometry* hcalBarrelGeo = geom.getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const CaloSubdetectorGeometry* hcalEndcapGeo = geom.getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

  const std::vector<DetId>& validBarrelDetIds = hcalBarrelGeo->getValidDetIds(DetId::Hcal, HcalBarrel);
  const std::vector<DetId>& validEndcapDetIds = hcalEndcapGeo->getValidDetIds(DetId::Hcal, HcalEndcap);
  std::vector<DetId> vecHcal;
  vecHcal.insert(vecHcal.end(), validBarrelDetIds.begin(), validBarrelDetIds.end());
  vecHcal.insert(vecHcal.end(), validEndcapDetIds.begin(), validEndcapDetIds.end());

  std::cout << "HBHETopologyGPU test " << validBarrelDetIds.size() << " " << validEndcapDetIds.size() << std::endl;
  std::cout << "HBHETopologyGPU test " << vecHcal.size() << std::endl;
  std::cout << "HBHETopologyGPU constructor" << std::endl;

  //
  // Filling HCAL DenseID vectors
  //std::vector<unsigned int> vDenseIdHcal_;
  vDenseIdHcal_.clear(); // vector of DenseIds

  vDenseIdHcal_.reserve(vecHcal.size());
  for (auto hDetId : vecHcal) {
    vDenseIdHcal_.push_back(topo.detId2denseId(hDetId));
    std::cout << topo.detId2denseId(hDetId) << std::endl;
  }
  std::sort(vDenseIdHcal_.begin(), vDenseIdHcal_.end());

  //
  // Filling information to define arrays for all relevant HBHE DetIds
  denseIdHcalMax_ = *max_element(vDenseIdHcal_.begin(), vDenseIdHcal_.end());
  denseIdHcalMin_ = *min_element(vDenseIdHcal_.begin(), vDenseIdHcal_.end());
  std::cout << "denseIdHcalMin_ " << denseIdHcalMin_ << std::endl;
  std::cout << "denseIdHcalMax_ " << denseIdHcalMax_ << std::endl;
  const int denseIdOffset = denseIdHcalMin_;
  const int detIdArraySize = denseIdHcalMax_ - denseIdHcalMin_ + 1;

  //
  // Filling a vector of cell neighbours
  //std::vector<std::vector<DetId>> neighboursHcal_;
  neighboursHcal_.clear(); // vector of neighbors
  neighboursHcal_.resize(detIdArraySize);

    for (auto denseid : vDenseIdHcal_) {
      DetId N(0);
      DetId E(0);
      DetId S(0);
      DetId W(0);
      DetId NW(0);
      DetId NE(0);
      DetId SW(0);
      DetId SE(0);
      std::vector<DetId> neighbours(9, DetId(0));

      // the centre
      unsigned denseid_c = denseid;
      DetId detid_c = topo.denseId2detId(denseid_c);
      CaloNavigator<HcalDetId> navigator(detid_c, &topo);

      HcalDetId hid_c = HcalDetId(detid_c);

      // Using enum in Geometry/CaloTopology/interface/CaloDirection.h
      // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
      neighbours.at(NONE) = detid_c;

      navigator.home();
      E = navigator.east(); // smaller ieta values
      neighbours.at(EAST) = E;
      HcalDetId hid_e = HcalDetId(E);

      navigator.home();
      W = navigator.west(); // larger ieta values
      neighbours.at(WEST) = W;
      HcalDetId hid_w = HcalDetId(W);

      std::cout << "C,E,W: ("
		<< hid_c.ieta() << " " << hid_c.iphi() << " " << hid_c.depth() << ") ("
		<< hid_e.ieta() << " " << hid_e.iphi() << " " << hid_e.depth() << ") ("
		<< hid_w.ieta() << " " << hid_w.iphi() << " " << hid_w.depth() << ") " << std::endl;
      if (hid_e.ieta()>hid_c.ieta() && hid_e.ieta()!=0) std::cout << "WARNING1" << std::endl;
      if (hid_w.ieta()<hid_c.ieta() && hid_w.ieta()!=0) std::cout << "WARNING2" << std::endl;
      if (hid_e.depth()!=hid_c.depth() && hid_e.depth()!=0) std::cout << "WARNING3" << std::endl;
      if (hid_w.depth()!=hid_c.depth() && hid_w.depth()!=0) std::cout << "WARNING4" << std::endl;

      navigator.home();
      N = navigator.north(); // larger iphi values (except phi boundary)
      neighbours.at(NORTH) = N;
      HcalDetId hid_n = HcalDetId(N);

      navigator.home();
      S = navigator.south(); // smaller iphi values (except phi boundary)
      neighbours.at(SOUTH) = S;
      HcalDetId hid_s = HcalDetId(S);

      std::cout << "C,N,S: ("
		<< hid_c.ieta() << " " << hid_c.iphi() << " " << hid_c.depth() << ") ("
		<< hid_n.ieta() << " " << hid_n.iphi() << " " << hid_n.depth() << ") ("
		<< hid_s.ieta() << " " << hid_s.iphi() << " " << hid_s.depth() << ") " << std::endl;
      if (hid_s.iphi()>hid_c.iphi() && hid_s.iphi()<65) std::cout << "WARNING5" << std::endl;
      if (hid_n.iphi()<hid_c.iphi() && hid_n.iphi()>5) std::cout << "WARNING6" << std::endl;
      if (hid_s.depth()!=hid_c.depth() && hid_s.depth()!=0) std::cout << "WARNING7" << std::endl;
      if (hid_n.depth()!=hid_c.depth() && hid_n.depth()!=0) std::cout << "WARNING8" << std::endl;

      // Corners
      navigator.home();
      navigator.east();
      if (hid_c.ieta() > 0.) {  // positive eta: east -> move to smaller |ieta| (finner phi granularity) first
        if (E != DetId(0)) {
          // SE
          SE = navigator.south();
          neighbours.at(SOUTHEAST) = SE;
          // NE
          navigator.home();
          navigator.east();
          NE = navigator.north();
          neighbours.at(NORTHEAST) = NE;
        }
      }  // ieta<0 is handled later.

      navigator.home();
      navigator.west();
      if (hid_c.ieta() < 0.) {  // negative eta: west -> move to smaller |ieta| (finner phi granularity) first
        if (W != DetId(0)) {
          NW = navigator.north();
          neighbours.at(NORTHWEST) = NW;
          //
          navigator.home();
          navigator.west();
          SW = navigator.south();
          neighbours.at(SOUTHWEST) = SW;
        }
      }  // ieta>0 is handled later.

      navigator.home();
      navigator.north();
      if (N != DetId(0)) {
        if (hid_c.ieta() < 0.) {  // negative eta: move in phi first then move to east (coarser phi granularity)
          NE = navigator.east();
          neighbours.at(NORTHEAST) = NE;
        } else {  // positive eta: move in phi first then move to west (coarser phi granularity)
          NW = navigator.west();
          neighbours.at(NORTHWEST) = NW;
        }
      }

      navigator.home();
      navigator.south();
      if (S != DetId(0)) {
        if (hid_c.ieta() > 0.) {  // positive eta: move in phi first then move to west (coarser phi granularity)
          SW = navigator.west();
          neighbours.at(SOUTHWEST) = SW;
        } else {  // negative eta: move in phi first then move to east (coarser phi granularity)
          SE = navigator.east();
          neighbours.at(SOUTHEAST) = SE;
        }
      }

      unsigned index = getIdx(denseid_c);
      neighboursHcal_[index] = neighbours;

    }

    //
    // Check backward compatibility (does a neighbour of a channel have the channel as a neighbour?)
    //
    for (auto denseid : vDenseIdHcal_) {
      DetId detid = topo.denseId2detId(denseid);
      HcalDetId hid = HcalDetId(detid);
      if (detid == DetId(0)) {
	std::cout << "WARNING SHOULD NOT HAPPEN1" << std::endl;
        continue;
      }
      if (!validNeighbours(denseid)){
	std::cout << "WARNING SHOULD NOT HAPPEN2" << std::endl;
        continue;
      }
      std::vector<DetId> neighbours(9, DetId(0));
      unsigned index = getIdx(denseid);
      if (index >= neighboursHcal_.size()){
	std::cout << "WARNING SHOULD NOT HAPPEN3" << std::endl;
        continue;  // Skip if not found
      }
      neighbours = neighboursHcal_.at(index);

      //
      // Loop over neighbours
      int ineighbour = -1;
      for (auto neighbour : neighbours) {
        ineighbour++;
        if (neighbour == DetId(0))
          continue;
        std::vector<DetId> neighboursOfNeighbour(9, DetId(0));
        std::unordered_set<unsigned int> listOfNeighboursOfNeighbour;  // list of neighbours of neighbour
        unsigned denseidNeighbour = topo.detId2denseId(neighbour);
        if (!validNeighbours(denseidNeighbour))
          continue;
        if (getIdx(denseidNeighbour) >= neighboursHcal_.size())
          continue;
        neighboursOfNeighbour = neighboursHcal_.at(getIdx(denseidNeighbour));

        //
        // Loop over neighbours of neighbours
        for (auto neighbourOfNeighbour : neighboursOfNeighbour) {
          if (neighbourOfNeighbour == DetId(0))
            continue;
          unsigned denseidNeighbourOfNeighbour = (&topo)->detId2denseId(neighbourOfNeighbour);
          if (!validNeighbours(denseidNeighbourOfNeighbour))
            continue;
          listOfNeighboursOfNeighbour.insert(denseidNeighbourOfNeighbour);
        }

        //
        if (listOfNeighboursOfNeighbour.find(denseid) == listOfNeighboursOfNeighbour.end()) {
          // this neighbour is not backward compatible. ignore in the canse of HE phi segmentation change boundary
          if (hid.subdet() == HcalBarrel || hid.subdet() == HcalEndcap) {
            //         std::cout << "This neighbor does not have the original channel as its neighbor. Ignore: "
            //                << detid.det() << " " << hid.ieta() << " " << hid.iphi() << " " << hid.depth() << " "
            //                << neighbour.det() << " " << hidn.ieta() << " " << hidn.iphi() << " " << hidn.depth()
            //                << std::endl;
            neighboursHcal_[index][ineighbour] = DetId(0);
          }
        }
      }  // loop over neighbours
    }    // loop over vDenseIdHcal_

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

  std::cout << "detid size " <<  detId.size() << std::endl;

  for (auto denseid : vDenseIdHcal_) {

    DetId detid = topo.denseId2detId(denseid);
    HcalDetId hid = HcalDetId(detid);
    GlobalPoint pos;
    if (hid.subdet() == HcalBarrel) pos = hcalBarrelGeo->getGeometry(detid)->getPosition();
    else if (hid.subdet() == HcalEndcap) pos = hcalEndcapGeo->getGeometry(detid)->getPosition();
    else std::cout << "Invalid subdetector found for detId " << hid.rawId() << ": " << hid.subdet() << std::endl;

    //validDetIdPositions.emplace_back(pos);
    unsigned index = getIdx(denseid);
    detId[index] = (uint32_t)detid;
    position[index] = make_float3(pos.x(),pos.y(),pos.z());

    auto neigh = neighboursHcal_.at(index);

    for (uint32_t n = 0; n < 8; n++) {
      // cmssdt.cern.ch/lxr/source/RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h#0087
      // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
      // neighboursHcal_[centerIndex][0] is the rechit itself. Skip for neighbour array
      // If no neighbour exists in a direction, the value will be 0
      // Some neighbors from HF included! Need to test if these are included in the map!
      auto neighDetId = neigh[n + 1].rawId();
      if (neighDetId > 0
	  && (&topo)->detId2denseId(neighDetId)>=denseIdHcalMin_
	  && (&topo)->detId2denseId(neighDetId)<=denseIdHcalMax_) {
	neighbours[index * 8 + n] = getIdx(topo.detId2denseId(neighDetId));
      } else
	neighbours[index * 8 + n] = -1;
    }

  }

  //
  // KH - of course these are dummy
  auto const& detId2 = ps.getParameter<std::vector<uint32_t>>("pulseOffsets");
  auto const& neighbours2 = ps.getParameter<std::vector<int>>("pulseOffsets2");

  //
  // Fill variables for HostAllocator
  detId_.resize(detId.size());
  std::copy(detId.begin(), detId.end(), detId_.begin());
  neighbours_.resize(neighbours.size());
  std::copy(neighbours.begin(), neighbours.end(), neighbours_.begin());
  position_.resize(position.size());
  std::copy(position.begin(), position.end(), position_.begin());

}

HBHETopologyGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(detId));
  cudaCheck(cudaFree(neighbours));
  cudaCheck(cudaFree(position));
}

HBHETopologyGPU::Product const& HBHETopologyGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HBHETopologyGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        //cudaCheck(cudaMalloc((void**)&product.values, this->values_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.detId, this->detId_.size() * sizeof(uint32_t)));
        cudaCheck(cudaMalloc((void**)&product.neighbours, this->neighbours_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.position, this->position_.size() * sizeof(float3)));

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
        cudaCheck(cudaMemcpyAsync(product.position,
                                  this->position_.data(),
                                  this->position_.size() * sizeof(float3),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));

	// We want (1) detId_ and (2) neighbours_ transferred to devices

      });

  return product;
}

TYPELOOKUP_DATA_REG(HBHETopologyGPU);
