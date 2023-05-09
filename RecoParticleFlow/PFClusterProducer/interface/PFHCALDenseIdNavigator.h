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

      // Using enum in Geometry/CaloTopology/interface/CaloDirection.h
      // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
      neighbours.at(NONE) = detid_c;

      navigator.home();
      N = navigator.north();
      neighbours.at(NORTH) = N;
      if (N != DetId(0)) {
        NE = navigator.east();
      } else {
        navigator.home();
        E = navigator.east();
        NE = navigator.north();
      }
      neighbours.at(NORTHEAST) = NE;

      navigator.home();
      S = navigator.south();
      neighbours.at(SOUTH) = S;
      if (S != DetId(0)) {
        SW = navigator.west();
      } else {
        navigator.home();
        W = navigator.west();
        SW = navigator.south();
      }
      neighbours.at(SOUTHWEST) = SW;

      navigator.home();
      E = navigator.east();
      neighbours.at(EAST) = E;
      if (E != DetId(0)) {
        SE = navigator.south();
      } else {
        navigator.home();
        S = navigator.south();
        SE = navigator.east();
      }
      neighbours.at(SOUTHEAST) = SE;

      navigator.home();
      W = navigator.west();
      neighbours.at(WEST) = W;
      if (W != DetId(0)) {
        NW = navigator.north();
      } else {
        navigator.home();
        N = navigator.north();
        NW = navigator.west();
      }
      neighbours.at(NORTHWEST) = NW;

      unsigned index = getIdx(denseid_c);
      neighboursHcal_[index] = neighbours;
    }
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
    if (neighboursHcal_.at(index).size() != 9)
      ok = false;  // the neighbour vector size should be 3x3
    return ok;
  }

  unsigned int getIdx(const unsigned int denseid) const {
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
