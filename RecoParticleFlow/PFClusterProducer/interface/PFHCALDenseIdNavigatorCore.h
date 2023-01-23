#ifndef RecoParticleFlow_PFClusterProducer_interface_PFHCALDenseIdNavigatorCore_h
#define RecoParticleFlow_PFClusterProducer_interface_PFHCALDenseIdNavigatorCore_h

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include <unordered_set>

//----------
class PFHCALDenseIdNavigatorCore{
public:
  ~PFHCALDenseIdNavigatorCore() {
  }

  PFHCALDenseIdNavigatorCore(const std::vector<int>& vhcalEnum,
			  const CaloGeometry& geom,
			  const HcalTopology& topo){

    //
    // Fill a vector of DetId of interest
    std::vector<DetId> vecHcal;
    for (auto hcalSubdet : vhcalEnum) {
      std::vector<DetId> vecDetIds(geom.getValidDetIds(DetId::Hcal, hcalSubdet));
      vecHcal.insert(vecHcal.end(), vecDetIds.begin(), vecDetIds.end());
    }
    std::cout << vecHcal.size() << std::endl;

    //
    // Filling HCAL DenseID vectors
    denseId_.clear(); // vector of DenseIds
    denseId_.reserve(vecHcal.size());
    for (auto hDetId : vecHcal) {
      denseId_.push_back(topo.detId2denseId(hDetId));
    }
    std::sort(denseId_.begin(), denseId_.end());

    //
    // Filling information to define arrays for all relevant HBHE DetIds
    denseIdMax_ = *max_element(denseId_.begin(), denseId_.end());
    denseIdMin_ = *min_element(denseId_.begin(), denseId_.end());
    const int detIdArraySize = denseIdMax_ - denseIdMin_ + 1;

    //
    // Filling a vector of cell neighbours
    neighbours_.clear(); // vector of neighbors
    neighbours_.resize(detIdArraySize);

    for (auto denseid : denseId_) {
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

      navigator.home();
      W = navigator.west(); // larger ieta values
      neighbours.at(WEST) = W;

      navigator.home();
      N = navigator.north(); // larger iphi values (except phi boundary)
      neighbours.at(NORTH) = N;

      navigator.home();
      S = navigator.south(); // smaller iphi values (except phi boundary)
      neighbours.at(SOUTH) = S;

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
      neighbours_[index] = neighbours;

    }

    //
    // Check backward compatibility (does a neighbour of a channel have the channel as a neighbour?)
    //
    for (auto denseid : denseId_) {
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
      if (index >= neighbours_.size()){
	std::cout << "WARNING SHOULD NOT HAPPEN3" << std::endl;
	continue;  // Skip if not found
      }
      neighbours = neighbours_.at(index);

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
	if (getIdx(denseidNeighbour) >= neighbours_.size())
	  continue;
	neighboursOfNeighbour = neighbours_.at(getIdx(denseidNeighbour));

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
	    neighbours_[index][ineighbour] = DetId(0);
	  }
	}
      }  // loop over neighbours
    }    // loop over denseId_

  }

  bool validNeighbours(const unsigned int denseid) const {
    bool ok = true;
    unsigned index = getIdx(denseid);
    if (index >= neighbours_.size() || neighbours_.at(index).size() != 9)
      ok = false;  // the neighbour vector size should be 3x3
    return ok;
  }

  unsigned int getIdx(const unsigned int denseid) const {
    unsigned index = denseid - denseIdMin_;
    return index;
  }

  std::vector<DetId> getNeighbours(const unsigned int denseid) {
    if ( !std::binary_search(denseId_.begin(), denseId_.end(), denseid) )
      std::cout << "PFHCALDenseIdNavigatorCore: Cannot find neighbour information of this denseId channel. denseId= " << denseid << std::endl;
    return neighbours_[getIdx(denseid)];
  }
  std::vector<std::vector<DetId>> getNeighboursList() { return neighbours_; }
  std::vector<unsigned int> getValidDenseIds() { return denseId_; }

protected:
  std::vector<int> vhcalEnum_;
  std::vector<unsigned int> denseId_;
  std::vector<std::vector<DetId>> neighbours_; // indexed based on denseId
  unsigned int denseIdMax_;
  unsigned int denseIdMin_;

private:
};

#endif  // RecoParticleFlow_PFClusterProducer_interface_PFHCALDenseIdNavigatorCore_h
