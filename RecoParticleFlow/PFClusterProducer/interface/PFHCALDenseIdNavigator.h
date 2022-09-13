#ifndef RecoParticleFlow_PFClusterProducer_interface_PFHCALDenseIdNavigator_h
#define RecoParticleFlow_PFClusterProducer_interface_PFHCALDenseIdNavigator_h

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
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"

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
        hcalToken_(cc.esConsumes<edm::Transition::BeginLuminosityBlock>()),
        geomToken_(cc.esConsumes<edm::Transition::BeginLuminosityBlock>()) {}

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
    vDenseIdHcal_.clear();
    neighboursHcal_.clear();
    for (auto hcalSubdet : vhcalEnum_) {
      std::vector<DetId> vecDetIds(caloGeom.getValidDetIds(DetId::Hcal, hcalSubdet));
      vecHcal.insert(vecHcal.end(), vecDetIds.begin(), vecDetIds.end());
    }
    vDenseIdHcal_.reserve(vecHcal.size());

    for (auto hDetId : vecHcal) {
      vDenseIdHcal_.push_back(topology_.get()->detId2denseId(hDetId));
    }
    std::sort(vDenseIdHcal_.begin(), vDenseIdHcal_.end());

    // Fill a vector of cell neighbours
    denseIdHcalMax_ = *max_element(vDenseIdHcal_.begin(), vDenseIdHcal_.end());
    denseIdHcalMin_ = *min_element(vDenseIdHcal_.begin(), vDenseIdHcal_.end());
    neighboursHcal_.resize(denseIdHcalMax_ - denseIdHcalMin_ + 1);
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
      DetId detid_c = topology_.get()->denseId2detId(denseid_c);
      CaloNavigator<DET> navigator(detid_c, topology_.get());

      HcalDetId hid_c = HcalDetId(detid_c);

      // Using enum in Geometry/CaloTopology/interface/CaloDirection.h
      // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
      neighbours.at(NONE) = detid_c;

      navigator.home();
      E = navigator.east();
      neighbours.at(EAST) = E;
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
      W = navigator.west();
      neighbours.at(WEST) = W;
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
      N = navigator.north();
      neighbours.at(NORTH) = N;
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
      S = navigator.south();
      neighbours.at(SOUTH) = S;
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
      DetId detid = topology_.get()->denseId2detId(denseid);
      HcalDetId hid = HcalDetId(detid);
      if (detid == DetId(0))
        continue;
      if (!validNeighbours(denseid))
        continue;
      std::vector<DetId> neighbours(9, DetId(0));
      unsigned index = getIdx(denseid);
      if (index >= neighboursHcal_.size())
        continue;  // Skip if not found
      neighbours = neighboursHcal_.at(index);

      //
      // Loop over neighbours
      int ineighbour = -1;
      for (auto neighbour : neighbours) {
        ineighbour++;
        if (neighbour == DetId(0))
          continue;
        //HcalDetId hidn  = HcalDetId(neighbour);
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
            //         std::cout << "This neighbor does not have the original channel as its neighbor. Ignore: "
            //                << detid.det() << " " << hid.ieta() << " " << hid.iphi() << " " << hid.depth() << " "
            //                << neighbour.det() << " " << hidn.ieta() << " " << hidn.iphi() << " " << hidn.depth()
            //                << std::endl;
            neighboursHcal_[index][ineighbour] = DetId(0);
          }
        }
      }  // loop over neighbours
    }    // loop over vDenseIdHcal_
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

  std::vector<DetId> getNeighbours(const unsigned int denseid) { return neighboursHcal_[getIdx(denseid)]; }

  std::vector<unsigned int>* getValidDenseIds() { return &vDenseIdHcal_; }

protected:
  edm::ESWatcher<HcalRecNumberingRecord> theRecNumberWatcher_;
  std::unique_ptr<const TOPO> topology_;
  std::vector<int> vhcalEnum_;
  std::vector<unsigned int> vDenseIdHcal_;
  std::vector<std::vector<DetId>> neighboursHcal_;
  unsigned int denseIdHcalMax_;
  unsigned int denseIdHcalMin_;

private:
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

#endif  // RecoParticleFlow_PFClusterProducer_interface_PFHCALDenseIdNavigator_h
