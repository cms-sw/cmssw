#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitHCALCachedNavigator_h
#define RecoParticleFlow_PFClusterProducer_PFRecHitHCALCachedNavigator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

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
class PFRecHitHCALCachedNavigator : public PFRecHitNavigatorBase {
public:
  ~PFRecHitHCALCachedNavigator() override {
    if (!ownsTopo) {
      topology_.release();
    }
  }

  PFRecHitHCALCachedNavigator(const edm::ParameterSet& iConfig) {
    vdetectorEnum_ = iConfig.getParameter<std::vector<int>>("detectorEnums");
  }

  void init(const edm::EventSetup& iSetup) override {
    if (!neighboursHcal_.empty())
      return;  // neighboursHcal_ is already defined. No need to redefine it.

    edm::ESHandle<HcalTopology> hcalTopology;
    iSetup.get<HcalRecNumberingRecord>().get(hcalTopology);
    topology_.release();
    topology_.reset(hcalTopology.product());

    // Fill a vector of valid denseid's
    edm::ESHandle<CaloGeometry> hGeom;
    iSetup.get<CaloGeometryRecord>().get(hGeom);
    const CaloGeometry& caloGeom = *hGeom;

    std::vector<DetId> vecHcal;
    std::vector<unsigned int> vDenseIdHcal;
    vecHcal.clear();
    vDenseIdHcal.clear();
    neighboursHcal_.clear();
    for (unsigned i = 0; i < vdetectorEnum_.size(); ++i) {
      std::vector<DetId> vecDetIds(caloGeom.getValidDetIds(DetId::Hcal, vdetectorEnum_[i]));
      vecHcal.insert(vecHcal.end(), vecDetIds.begin(), vecDetIds.end());
    }
    for (unsigned i = 0; i < vecHcal.size(); ++i) {
      vDenseIdHcal.push_back(topology_.get()->detId2denseId(vecHcal[i]));
    }
    std::sort(vDenseIdHcal.begin(), vDenseIdHcal.end());

    // Fill a vector of cell neighbours
    denseIdHcalMax_ = *max_element(vDenseIdHcal.begin(), vDenseIdHcal.end());
    denseIdHcalMin_ = *min_element(vDenseIdHcal.begin(), vDenseIdHcal.end());
    neighboursHcal_.resize(denseIdHcalMax_ - denseIdHcalMin_ + 1);

    for (unsigned i = 0; i < vDenseIdHcal.size(); ++i) {
      DetId N(0);
      DetId E(0);
      DetId S(0);
      DetId W(0);
      DetId NW(0);
      DetId NE(0);
      DetId SW(0);
      DetId SE(0);
      std::vector<DetId> neighbours;
      neighbours.resize(9);

      // the centre
      unsigned denseid_c = vDenseIdHcal[i];
      DetId detid_c = topology_.get()->denseId2detId(denseid_c);
      CaloNavigator<DET> navigator(detid_c, topology_.get());

      // Using enum in Geometry/CaloTopology/interface/CaloDirection.h
      // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
      neighbours[NONE] = detid_c;

      navigator.home();
      N = navigator.north();
      neighbours[NORTH] = N;
      if (N != DetId(0)) {
        NE = navigator.east();
      } else {
        navigator.home();
        E = navigator.east();
        NE = navigator.north();
      }
      neighbours[NORTHEAST] = NE;

      navigator.home();
      S = navigator.south();
      neighbours[SOUTH] = S;
      if (S != DetId(0)) {
        SW = navigator.west();
      } else {
        navigator.home();
        W = navigator.west();
        SW = navigator.south();
      }
      neighbours[SOUTHWEST] = SW;

      navigator.home();
      E = navigator.east();
      neighbours[EAST] = E;
      if (E != DetId(0)) {
        SE = navigator.south();
      } else {
        navigator.home();
        S = navigator.south();
        SE = navigator.east();
      }
      neighbours[SOUTHEAST] = SE;

      navigator.home();
      W = navigator.west();
      neighbours[WEST] = W;
      if (W != DetId(0)) {
        NW = navigator.north();
      } else {
        navigator.home();
        N = navigator.north();
        NW = navigator.west();
      }
      neighbours[NORTHWEST] = NW;

      unsigned index = denseid_c - denseIdHcalMin_;
      neighboursHcal_[index] = neighbours;
    }
  }

  void associateNeighbours(reco::PFRecHit& hit,
                           std::unique_ptr<reco::PFRecHitCollection>& hits,
                           edm::RefProd<reco::PFRecHitCollection>& refProd) override {
    DetId detid(hit.detId());
    unsigned denseid = topology_.get()->detId2denseId(detid);

    if (denseid < denseIdHcalMin_ || denseid > denseIdHcalMax_) {
      edm::LogWarning("PFRecHitHCALCachedNavigator") << " DenseId for this cell is out of the range." << std::endl;
    } else if (!validNeighbours(denseid)) {
      edm::LogWarning("PFRecHitHCALCachedNavigator")
          << " DenseId for this cell does not have the neighbour information." << std::endl;
    } else {
      unsigned index = denseid - denseIdHcalMin_;
      associateNeighbour(neighboursHcal_[index][NORTH], hit, hits, refProd, 0, 1, 0);        // N
      associateNeighbour(neighboursHcal_[index][NORTHEAST], hit, hits, refProd, 1, 1, 0);    // NE
      associateNeighbour(neighboursHcal_[index][SOUTH], hit, hits, refProd, 0, -1, 0);       // S
      associateNeighbour(neighboursHcal_[index][SOUTHWEST], hit, hits, refProd, -1, -1, 0);  // SW
      associateNeighbour(neighboursHcal_[index][EAST], hit, hits, refProd, 1, 0, 0);         // E
      associateNeighbour(neighboursHcal_[index][SOUTHEAST], hit, hits, refProd, 1, -1, 0);   // SE
      associateNeighbour(neighboursHcal_[index][WEST], hit, hits, refProd, -1, 0, 0);        // W
      associateNeighbour(neighboursHcal_[index][NORTHWEST], hit, hits, refProd, -1, 1, 0);   // NW
    }
  }

  bool validNeighbours(const unsigned int denseid) const {
    bool ok = true;
    unsigned index = denseid - denseIdHcalMin_;
    if (neighboursHcal_[index].size() != 9)
      ok = false;  // the neighbour vector size should be 3x3
    return ok;
  }

protected:
  std::unique_ptr<const TOPO> topology_;
  std::vector<int> vdetectorEnum_;
  std::vector<std::vector<DetId>> neighboursHcal_;
  unsigned int denseIdHcalMax_;
  unsigned int denseIdHcalMin_;
};

#endif
