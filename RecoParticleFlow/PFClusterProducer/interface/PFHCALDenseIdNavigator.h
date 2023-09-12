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

  // Initialization
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

    }  // for denseid vDenseIdHcal

    if (debug) {
      backwardCompatibilityCheck(vDenseIdHcal);
      printNeighbourInfo(vDenseIdHcal);
    }
  }

  // Check neighbour
  void backwardCompatibilityCheck(const std::vector<unsigned int> vDenseIdHcal) {
    //
    // Check backward compatibility (does a neighbour of a channel have the channel as a neighbour?)
    //
    for (auto denseid : vDenseIdHcal) {
      DetId detid = topology_.get()->denseId2detId(denseid);
      HcalDetId hid = HcalDetId(detid);
      if (detid == DetId(0)) {
        edm::LogWarning("PFHCALDenseIdNavigator") << "Found an invalid DetId";
        continue;
      }
      if (!validNeighbours(denseid)) {
        edm::LogWarning("PFHCALDenseIdNavigator") << "The vector for neighbour information has an invalid length";
        continue;
      }
      std::vector<DetId> neighbours(9, DetId(0));
      unsigned index = getIdx(denseid);
      if (index >= neighboursHcal_.size()) {
        edm::LogWarning("PFHCALDenseIdNavigator") << "The vector for neighbour information is not found";
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

        if (!validNeighbours(denseidNeighbour)) {
          edm::LogWarning("PFHCALDenseIdNavigator")
              << "This neighbour does not have a valid neighbour information (another subdetector?). Ignore: "
              << detid.det() << " " << hid.ieta() << " " << hid.iphi() << " " << hid.depth() << " " << neighbour.det()
              << " " << hidn.ieta() << " " << hidn.iphi() << " " << hidn.depth();
          neighboursHcal_[index][ineighbour] = DetId(0);  // not a valid neighbour. set to a DetId(0)
          continue;
        }
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
          // This neighbour is not backward compatible.
          edm::LogWarning("PFHCALDenseIdNavigator")
              << "This neighbour does not have the original channel as its neighbour. Ignore: " << detid.det() << " "
              << hid.ieta() << " " << hid.iphi() << " " << hid.depth() << " " << neighbour.det() << " " << hidn.ieta()
              << " " << hidn.iphi() << " " << hidn.depth();
          neighboursHcal_[index][ineighbour] = DetId(0);
        }

      }  // loop over neighbours
    }    // loop over denseId_
  }

  // Print out neighbour DetId's
  void printNeighbourInfo(const std::vector<unsigned int> vDenseIdHcal) {
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

      const DetId NE = neighbours.at(NORTHEAST);
      const DetId SW = neighbours.at(SOUTHWEST);
      const DetId SE = neighbours.at(SOUTHEAST);
      const DetId NW = neighbours.at(NORTHWEST);

      edm::LogPrint("PFHCALDenseIdNavigator")
          << "PFHCALDenseIdNavigator: " << HcalDetId(detid_c).ieta() << " " << HcalDetId(detid_c).iphi() << " "
          << HcalDetId(detid_c).depth() << " " << HcalDetId(detid_c).subdetId() << ": " << HcalDetId(N).ieta() << " "
          << HcalDetId(N).iphi() << " " << HcalDetId(N).depth() << " " << HcalDetId(N).subdetId() << ", "
          << HcalDetId(E).ieta() << " " << HcalDetId(E).iphi() << " " << HcalDetId(E).depth() << " "
          << HcalDetId(E).subdetId() << ", " << HcalDetId(S).ieta() << " " << HcalDetId(S).iphi() << " "
          << HcalDetId(S).depth() << " " << HcalDetId(S).subdetId() << ", " << HcalDetId(W).ieta() << " "
          << HcalDetId(W).iphi() << " " << HcalDetId(W).depth() << " " << HcalDetId(W).subdetId() << ", "
          << HcalDetId(NE).ieta() << " " << HcalDetId(NE).iphi() << " " << HcalDetId(NE).depth() << " "
          << HcalDetId(NE).subdetId() << ", " << HcalDetId(SW).ieta() << " " << HcalDetId(SW).iphi() << " "
          << HcalDetId(SW).depth() << " " << HcalDetId(SW).subdetId() << ", " << HcalDetId(SE).ieta() << " "
          << HcalDetId(SE).iphi() << " " << HcalDetId(SE).depth() << " " << HcalDetId(SE).subdetId() << ", "
          << HcalDetId(NW).ieta() << " " << HcalDetId(NW).iphi() << " " << HcalDetId(NW).depth() << " "
          << HcalDetId(NW).subdetId();

    }  // print ends
  }

  // Check if two DetId's have the same subdetId
  const bool checkSameSubDet(const DetId detId, const DetId detId2) {
    HcalDetId hid = HcalDetId(detId);
    HcalDetId hid2 = HcalDetId(detId2);
    return hid.subdetId() == hid2.subdetId();
  }

  //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0141
  static constexpr int getZside(const DetId detId) { return ((detId & HcalDetId::kHcalZsideMask2) ? (1) : (-1)); }

  // Obtain the neighbour's DetId
  const uint32_t getNeighbourDetId(const DetId detId, const uint32_t direction) {
    // desired order for PF: NORTH, SOUTH, EAST, WEST, NORTHEAST, SOUTHWEST, SOUTHEAST, NORTHWEST
    if (detId == 0)
      return 0;

    if (direction == 0)                        // NORTH
      return topology_.get()->goNorth(detId);  // larger iphi values (except phi boundary)

    if (direction == 1)                        // SOUTH
      return topology_.get()->goSouth(detId);  // smaller iphi values (except phi boundary)

    // In the case of East/West, make sure we are not moving to another subdetector
    if (direction == 2 && checkSameSubDet(detId, topology_.get()->goEast(detId)))  // EAST
      return topology_.get()->goEast(detId);                                       // smaller ieta values

    if (direction == 3 && checkSameSubDet(detId, topology_.get()->goWest(detId)))  // WEST
      return topology_.get()->goWest(detId);                                       // larger ieta values

    // In the case of cornor cells, ensure backward compatibility.
    // Also make sure it's not already defined as E(ast) or W(est)
    if (direction == 4) {         // NORTHEAST
      if (getZside(detId) > 0) {  // positive eta: east -> move to smaller |ieta| (finner phi granularity) first
        const DetId E = getNeighbourDetId(detId, 2);
        const DetId NE = getNeighbourDetId(E, 0);
        if (getNeighbourDetId(NE, 1) == E && NE != E)
          return NE;  //
      } else {        // negative eta: move in phi first then move to east (coarser phi granularity)
        const DetId N = getNeighbourDetId(detId, 0);
        const DetId NE = getNeighbourDetId(N, 2);
        const DetId E = getNeighbourDetId(detId, 2);
        if (getNeighbourDetId(NE, 3) == N && NE != E)
          return NE;
      }
    }
    if (direction == 5) {         // SOUTHWEST
      if (getZside(detId) > 0) {  // positive eta: move in phi first then move to west (coarser phi granularity)
        const DetId S = getNeighbourDetId(detId, 1);
        const DetId SW = getNeighbourDetId(S, 3);
        const DetId W = getNeighbourDetId(detId, 3);
        if (getNeighbourDetId(SW, 2) == S && SW != W)
          return SW;
      } else {  // negative eta: west -> move to smaller |ieta| (finner phi granularity) first
        const DetId W = getNeighbourDetId(detId, 3);
        const DetId SW = getNeighbourDetId(W, 1);
        if (getNeighbourDetId(SW, 0) == W && SW != W)
          return SW;
      }
    }
    if (direction == 6) {         // SOUTHEAST
      if (getZside(detId) > 0) {  // positive eta: east -> move to smaller |ieta| (finner phi granularity) first
        const DetId E = getNeighbourDetId(detId, 2);
        const DetId SE = getNeighbourDetId(E, 1);
        if (getNeighbourDetId(SE, 0) == E && SE != E)
          return SE;
      } else {  // negative eta: move in phi first then move to east (coarser phi granularity)
        const DetId S = getNeighbourDetId(detId, 1);
        const DetId SE = getNeighbourDetId(S, 2);
        const DetId E = getNeighbourDetId(detId, 2);
        if (getNeighbourDetId(SE, 3) == S && SE != E)
          return SE;
      }
    }
    if (direction == 7) {         // NORTHWEST
      if (getZside(detId) > 0) {  // positive eta: move in phi first then move to west (coarser phi granularity)
        const DetId N = getNeighbourDetId(detId, 0);
        const DetId NW = getNeighbourDetId(N, 3);
        const DetId W = getNeighbourDetId(detId, 3);
        if (getNeighbourDetId(NW, 2) == N && NW != W)
          return NW;
      } else {  // negative eta: west -> move to smaller |ieta| (finner phi granularity) first
        const DetId W = getNeighbourDetId(detId, 3);
        const DetId NW = getNeighbourDetId(W, 0);
        if (getNeighbourDetId(NW, 1) == W && NW != W)
          return NW;
      }
    }
    return 0;
  }

  // Associate neighbour PFRecHits
  void associateNeighbours(reco::PFRecHit& hit,
                           std::unique_ptr<reco::PFRecHitCollection>& hits,
                           edm::RefProd<reco::PFRecHitCollection>& refProd) override {
    DetId detid(hit.detId());
    HcalDetId hid(detid);
    unsigned denseid = topology_.get()->detId2denseId(detid);

    std::vector<DetId> neighbours(9, DetId(0));

    if (denseid < denseIdHcalMin_ || denseid > denseIdHcalMax_) {
      edm::LogWarning("PFRecHitHCALCachedNavigator")
          << " DenseId for this cell is out of the expected range." << std::endl;
    } else if (!validNeighbours(denseid)) {
      edm::LogWarning("PFRecHitHCALCachedNavigator")
          << " DenseId for this cell does not have the neighbour information. " << hid.ieta() << " " << hid.iphi()
          << " " << hid.depth() << " " << hid.subdetId();
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

  // Check if we can get the valid neighbour vector for a given denseid
  const bool validNeighbours(const unsigned int denseid) const {
    if (denseid < denseIdHcalMin_ || denseid > denseIdHcalMax_)
      return false;  // neighbour denseid is out of the expected range
    unsigned index = getIdx(denseid);
    if (neighboursHcal_.at(index).size() != 9)
      return false;  // the neighbour vector size should be 3x3
    return true;
  }

  // Get the index of neighboursHcal_ for a given denseid
  const unsigned int getIdx(const unsigned int denseid) const {
    if (denseid < denseIdHcalMin_ || denseid > denseIdHcalMax_)
      return unsigned(denseIdHcalMax_ - denseIdHcalMin_ +
                      1);  // out-of-bounce (different subdetector groups. give a dummy number.)
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
  const bool debug = false;
};

#endif
