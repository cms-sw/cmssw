#ifndef RecoParticleFlow_PFClusterProducer_PFHCALDenseIdNavigator_h
#define RecoParticleFlow_PFClusterProducer_PFHCALDenseIdNavigator_h

#include "FWCore/Framework/interface/ESWatcher.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

template <typename DET, typename TOPO, bool ownsTopo = true>
class PFHCALDenseIdNavigator : public PFRecHitNavigatorBase {
public:
  ~PFHCALDenseIdNavigator() override {
    if (!ownsTopo) {
      topology_.release();
    }
  }

  PFHCALDenseIdNavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : vhcalEnum_(iConfig.getParameter<std::vector<int>>("hcalEnums")),
        hcalToken_(cc.esConsumes<edm::Transition::BeginRun>()),
        geomToken_(cc.esConsumes<edm::Transition::BeginRun>()) {}

  void init(const edm::EventSetup& iSetup) override {
    bool check = theRecNumberWatcher_.check(iSetup);
    if (!check)
      return;

    edm::ESHandle<HcalTopology> hcalTopology = iSetup.getHandle(hcalToken_);
    topology_.release();
    topology_.reset(hcalTopology.product());

    // Fill a vector of valid denseid's
    edm::ESHandle<CaloGeometry> hGeom = iSetup.getHandle(geomToken_);
    const CaloGeometry& caloGeom = *hGeom;

    std::vector<DetId> vecHcal;
    std::vector<unsigned int> vDenseIdHcal;
    neighboursHcal_.clear();
    for (auto hcalSubdet : vhcalEnum_) {
      std::vector<DetId> vecDetIds(caloGeom.getValidDetIds(DetId::Hcal, hcalSubdet));
      vecHcal.insert(vecHcal.end(), vecDetIds.begin(), vecDetIds.end());
    }
    vDenseIdHcal.reserve(vecHcal.size());
    for (auto hDetId : vecHcal) {
      vDenseIdHcal.push_back(topology_.get()->detId2denseId(hDetId));
    }
    std::sort(vDenseIdHcal.begin(), vDenseIdHcal.end());

    // Fill a vector of cell neighbours
    denseIdHcalMax_ = *max_element(vDenseIdHcal.begin(), vDenseIdHcal.end());
    denseIdHcalMin_ = *min_element(vDenseIdHcal.begin(), vDenseIdHcal.end());
    neighboursHcal_.resize(denseIdHcalMax_ - denseIdHcalMin_ + 1);

    for (auto denseid : vDenseIdHcal) {
      
      std::vector<DetId> neighbours(9, DetId(0));

      // the centre
      unsigned denseid_c = denseid;
      DetId detid_c = topology_.get()->denseId2detId(denseid_c);
      CaloNavigator<DET> navigator(detid_c, topology_.get());

      // Using enum in Geometry/CaloTopology/interface/CaloDirection.h
      // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
      neighbours.at(NONE) = detid_c;

      neighbours.at(NORTH) = getNeighbourDetId(detid_c, 0);
      neighbours.at(SOUTH) = getNeighbourDetId(detid_c, 1);
      neighbours.at(EAST) = getNeighbourDetId(detid_c, 2);
      neighbours.at(WEST) = getNeighbourDetId(detid_c, 3);

      neighbours.at(NORTHEAST) = getNeighbourDetId(detid_c, 4);
      neighbours.at(SOUTHWEST) = getNeighbourDetId(detid_c, 5);
      neighbours.at(SOUTHEAST) = getNeighbourDetId(detid_c, 6);
      neighbours.at(NORTHWEST) = getNeighbourDetId(detid_c, 7);

      unsigned index = getIdx(denseid_c);
      neighboursHcal_[index] = neighbours;

    }

    //
    // Check backward compatibility (does a neighbour of a channel have the channel as a neighbour?)
    //
    for (auto denseid : vDenseIdHcal) {
      DetId detid = topology_.get()->denseId2detId(denseid);
      HcalDetId hid = HcalDetId(detid);
      if (detid == DetId(0)) {
        std::cout << "WARNING SHOULD NOT HAPPEN1" << std::endl;
        continue;
      }
      if (!validNeighbours(denseid)) {
        std::cout << "WARNING SHOULD NOT HAPPEN2" << std::endl;
        continue;
      }
      std::vector<DetId> neighbours(9, DetId(0));
      unsigned index = getIdx(denseid);
      if (index >= neighboursHcal_.size()) {
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
	HcalDetId hidn = HcalDetId(neighbour);
        std::vector<DetId> neighboursOfNeighbour(9, DetId(0));
        std::unordered_set<unsigned int> listOfNeighboursOfNeighbour;  // list of neighbours of neighbour
        unsigned denseidNeighbour = topology_.get()->detId2denseId(neighbour);

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
          unsigned denseidNeighbourOfNeighbour = topology_.get()->detId2denseId(neighbourOfNeighbour);
          if (!validNeighbours(denseidNeighbourOfNeighbour))
            continue;
          listOfNeighboursOfNeighbour.insert(denseidNeighbourOfNeighbour);
        }

        //
        if (listOfNeighboursOfNeighbour.find(denseid) == listOfNeighboursOfNeighbour.end()) {
          // this neighbour is not backward compatible. ignore in the canse of HE phi segmentation change boundary
          if (hid.subdet() == HcalBarrel || hid.subdet() == HcalEndcap) {
	    std::cout << "This neighbor does not have the original channel as its neighbor. Ignore: "
		      << detid.det() << " " << hid.ieta() << " " << hid.iphi() << " " << hid.depth() << " "
		      << neighbour.det() << " " << hidn.ieta() << " " << hidn.iphi() << " " << hidn.depth()
		      << std::endl;
            neighboursHcal_[index][ineighbour] = DetId(0);
          }
        }
      }  // loop over neighbours
    }    // loop over denseId_

    //
    // Print final neighbour definitions
    //
    for (auto denseid : vDenseIdHcal) {
      std::vector<DetId> neighbours(9, DetId(0));

      // the centre
      unsigned denseid_c = denseid;
      DetId detid_c = topology_.get()->denseId2detId(denseid_c);
      CaloNavigator<DET> navigator(detid_c, topology_.get());

      unsigned index = getIdx(denseid_c);
      neighbours = neighboursHcal_[index];

      const DetId N = neighbours.at(NORTH);
      const DetId S = neighbours.at(SOUTH);
      const DetId E = neighbours.at(EAST);
      const DetId W = neighbours.at(WEST);

      DetId NE = neighbours.at(NORTHEAST); if (NE==E && E!=DetId(0)){ NE = DetId(0); printf("Duplicate A\n");}
      DetId SW = neighbours.at(SOUTHWEST); if (SW==W && W!=DetId(0)){ SW = DetId(0); printf("Duplicate B\n");}
      DetId SE = neighbours.at(SOUTHEAST); if (SE==E && E!=DetId(0)){ SE = DetId(0); printf("Duplicate C\n");}
      DetId NW = neighbours.at(NORTHWEST); if (NW==W && W!=DetId(0)){ NW = DetId(0); printf("Duplicate D\n");}
      
      printf("navi: %d %d %d %d: %d %d %d %d, %d %d %d %d, %d %d %d %d, %d %d %d %d:  %d %d %d %d, %d %d %d %d, %d %d %d %d, %d %d %d %d\n",
	     HcalDetId(detid_c).ieta(),HcalDetId(detid_c).iphi(),HcalDetId(detid_c).depth(),HcalDetId(detid_c).subdetId(),
	     HcalDetId(N).ieta(),HcalDetId(N).iphi(),HcalDetId(N).depth(),HcalDetId(N).subdetId(),
	     HcalDetId(E).ieta(),HcalDetId(E).iphi(),HcalDetId(E).depth(),HcalDetId(E).subdetId(),
	     HcalDetId(S).ieta(),HcalDetId(S).iphi(),HcalDetId(S).depth(),HcalDetId(S).subdetId(),
	     HcalDetId(W).ieta(),HcalDetId(W).iphi(),HcalDetId(W).depth(),HcalDetId(W).subdetId(),
	     HcalDetId(NE).ieta(),HcalDetId(NE).iphi(),HcalDetId(NE).depth(),HcalDetId(NE).subdetId(),
	     HcalDetId(SW).ieta(),HcalDetId(SW).iphi(),HcalDetId(SW).depth(),HcalDetId(SW).subdetId(),
	     HcalDetId(SE).ieta(),HcalDetId(SE).iphi(),HcalDetId(SE).depth(),HcalDetId(SE).subdetId(),
	     HcalDetId(NW).ieta(),HcalDetId(NW).iphi(),HcalDetId(NW).depth(),HcalDetId(NW).subdetId()
	     );
    } // print ends

  }

  //
  const bool checkSameSubDet(const DetId detId, const DetId detId2)
  {
    HcalDetId hid = HcalDetId(detId);
    HcalDetId hid2 = HcalDetId(detId2);
    return hid.subdetId()==hid2.subdetId();
  }

  //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0141
  static constexpr int getZside(const DetId detId) {
    return ((detId & HcalDetId::kHcalZsideMask2) ? (1) : (-1));
  }

  //
  const uint32_t getNeighbourDetId(const DetId detId, const uint32_t direction) {

    // desired order for PF: NORTH, SOUTH, EAST, WEST, NORTHEAST, SOUTHWEST, SOUTHEAST, NORTHWEST
    if(detId == 0)
      return 0;

    if(direction == 0)  // NORTH
      return topology_.get()->goNorth(detId);  // larger iphi values (except phi boundary)

    if(direction == 1)  // SOUTH
      return topology_.get()->goSouth(detId);  // smaller iphi values (except phi boundary)

    if(direction == 2 && checkSameSubDet(detId, topology_.get()->goEast(detId)))  // EAST
      return topology_.get()->goEast(detId);  // smaller ieta values

    if(direction == 3 && checkSameSubDet(detId, topology_.get()->goWest(detId)))  // WEST
      return topology_.get()->goWest(detId);  // larger ieta values

    if(direction == 4) { // NORTHEAST
      if (getZside(detId) > 0){  // positive eta: east -> move to smaller |ieta| (finner phi granularity) first
        //return getNeighbourDetId(getNeighbourDetId(detId, 0), 2);
	const DetId E = getNeighbourDetId(detId, 2);
	const DetId NE = getNeighbourDetId(E, 0);
	if (getNeighbourDetId(NE, 1) == E && NE != E) return NE; //
      }
      else { // negative eta: move in phi first then move to east (coarser phi granularity)
        //return getNeighbourDetId(getNeighbourDetId(detId, 0), 2);
 	const DetId N = getNeighbourDetId(detId, 0);
	const DetId NE = getNeighbourDetId(N, 2);
	const DetId E = getNeighbourDetId(detId, 2);
	if (getNeighbourDetId(NE, 3) == N && NE != E) return NE;
      }
    }
    if(direction == 5) {  // SOUTHWEST
      if (getZside(detId) > 0){  // positive eta: move in phi first then move to west (coarser phi granularity)
        //return getNeighbourDetId(getNeighbourDetId(detId, 1), 3);
	const DetId S = getNeighbourDetId(detId, 1);
	const DetId SW = getNeighbourDetId(S, 3);
	const DetId W = getNeighbourDetId(detId, 3);
	if (getNeighbourDetId(SW, 2) == S && SW != W) return SW;
      }
      else { // negative eta: west -> move to smaller |ieta| (finner phi granularity) first
        //return getNeighbourDetId(getNeighbourDetId(detId, 3), 1);
	const DetId W = getNeighbourDetId(detId, 3);
	const DetId SW = getNeighbourDetId(W, 1);
	if (getNeighbourDetId(SW, 0) == W && SW != W) return SW;
      }
    }
    if(direction == 6) {  // SOUTHEAST
      if (getZside(detId) > 0) { // positive eta: east -> move to smaller |ieta| (finner phi granularity) first
        //return getNeighbourDetId(getNeighbourDetId(detId, 2), 1);
	const DetId E = getNeighbourDetId(detId, 2);
	const DetId SE = getNeighbourDetId(E, 1);
	if (getNeighbourDetId(SE, 0) == E && SE != E) return SE;
      }
      else { // negative eta: move in phi first then move to east (coarser phi granularity)
        //return getNeighbourDetId(getNeighbourDetId(detId, 1), 2);
	const DetId S = getNeighbourDetId(detId, 1);
	const DetId SE = getNeighbourDetId(S, 2);
	const DetId E = getNeighbourDetId(detId, 2);
	if (getNeighbourDetId(SE, 3) == S && SE != E) return SE;
      }
    }
    if(direction == 7) {  // NORTHWEST
      if (getZside(detId) > 0) { // positive eta: move in phi first then move to west (coarser phi granularity)
        //return getNeighbourDetId(getNeighbourDetId(detId, 0), 3);
	const DetId N = getNeighbourDetId(detId, 0);
	const DetId NW = getNeighbourDetId(N, 3);
	const DetId W = getNeighbourDetId(detId, 3);
	if (getNeighbourDetId(NW, 2) == N && NW != W) return NW;
      }
      else {  // negative eta: west -> move to smaller |ieta| (finner phi granularity) first
        //return getNeighbourDetId(getNeighbourDetId(detId, 3), 0);
	const DetId W = getNeighbourDetId(detId, 3);
	const DetId NW = getNeighbourDetId(W, 0);
	if (getNeighbourDetId(NW, 1) == W && NW != W) return NW;
      }
    }
    return 0;

  }

  void associateNeighbours(reco::PFRecHit& hit,
                           std::unique_ptr<reco::PFRecHitCollection>& hits,
                           edm::RefProd<reco::PFRecHitCollection>& refProd) override {
    DetId detid(hit.detId());
    unsigned denseid = topology_.get()->detId2denseId(detid);

    std::vector<DetId> neighbours(9, DetId(0));

    if (denseid < denseIdHcalMin_ || denseid > denseIdHcalMax_) {
      edm::LogWarning("PFRecHitHCALCachedNavigator") << " DenseId for this cell is out of the range." << std::endl;
    } else if (!validNeighbours(denseid)) {
      edm::LogWarning("PFRecHitHCALCachedNavigator")
          << " DenseId for this cell does not have the neighbour information." << std::endl;
    } else {
      unsigned index = getIdx(denseid);
      neighbours = neighboursHcal_.at(index);
    }

    associateNeighbour(neighbours.at(NORTH), hit, hits, refProd, 0, 1, 0);        // N
    associateNeighbour(neighbours.at(NORTHEAST), hit, hits, refProd, 1, 1, 0);    // NE
    associateNeighbour(neighbours.at(SOUTH), hit, hits, refProd, 0, -1, 0);       // S
    associateNeighbour(neighbours.at(SOUTHWEST), hit, hits, refProd, -1, -1, 0);  // SW
    associateNeighbour(neighbours.at(EAST), hit, hits, refProd, 1, 0, 0);         // E
    associateNeighbour(neighbours.at(SOUTHEAST), hit, hits, refProd, 1, -1, 0);   // SE
    associateNeighbour(neighbours.at(WEST), hit, hits, refProd, -1, 0, 0);        // W
    associateNeighbour(neighbours.at(NORTHWEST), hit, hits, refProd, -1, 1, 0);   // NW
  }

  const bool validNeighbours(const unsigned int denseid) const {
    bool ok = true;
    unsigned index = getIdx(denseid);
    if (neighboursHcal_.at(index).size() != 9)
      ok = false;  // the neighbour vector size should be 3x3
    return ok;
  }

  const unsigned int getIdx(const unsigned int denseid) const {
    unsigned index = denseid - denseIdHcalMin_;
    return index;
  }

protected:
  edm::ESWatcher<HcalRecNumberingRecord> theRecNumberWatcher_;
  std::unique_ptr<const TOPO> topology_;
  std::vector<int> vhcalEnum_;
  std::vector<std::vector<DetId>> neighboursHcal_;
  unsigned int denseIdHcalMax_;
  unsigned int denseIdHcalMin_;

private:
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

#endif
