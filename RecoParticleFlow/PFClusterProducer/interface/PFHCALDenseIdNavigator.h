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
#include "RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigatorCore.h"

//----------
template <typename DET, typename TOPO, bool ownsTopo = true>
class PFHCALDenseIdNavigator : public PFRecHitNavigatorBase {
public:
  ~PFHCALDenseIdNavigator() override {
    if (!ownsTopo) {
      topology_.release();
      navicore_.release();
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

    // Utilize PFHCALDenseIdNavigatorCore
    navicore_ = std::make_unique<PFHCALDenseIdNavigatorCore>(vhcalEnum_, caloGeom, *topology_.get());
    vDenseIdHcal_.clear();
    vDenseIdHcal_ = navicore_.get()->getValidDenseIds();
  }

  void associateNeighbours(reco::PFRecHit& hit,
                           std::unique_ptr<reco::PFRecHitCollection>& hits,
                           edm::RefProd<reco::PFRecHitCollection>& refProd) override {
    DetId detid(hit.detId());
    auto denseid = topology_.get()->detId2denseId(detid);
    auto neighbours = navicore_.get()->getNeighbours(denseid);

    associateNeighbour(neighbours.at(NORTH), hit, hits, refProd, 0, 1, 0);        // N
    associateNeighbour(neighbours.at(NORTHEAST), hit, hits, refProd, 1, 1, 0);    // NE
    associateNeighbour(neighbours.at(SOUTH), hit, hits, refProd, 0, -1, 0);       // S
    associateNeighbour(neighbours.at(SOUTHWEST), hit, hits, refProd, -1, -1, 0);  // SW
    associateNeighbour(neighbours.at(EAST), hit, hits, refProd, 1, 0, 0);         // E
    associateNeighbour(neighbours.at(SOUTHEAST), hit, hits, refProd, 1, -1, 0);   // SE
    associateNeighbour(neighbours.at(WEST), hit, hits, refProd, -1, 0, 0);        // W
    associateNeighbour(neighbours.at(NORTHWEST), hit, hits, refProd, -1, 1, 0);   // NW
  }

  std::vector<unsigned int>* getValidDenseIds() { return &vDenseIdHcal_; }

protected:
  edm::ESWatcher<HcalRecNumberingRecord> theRecNumberWatcher_;
  std::unique_ptr<const TOPO> topology_;
  std::unique_ptr<PFHCALDenseIdNavigatorCore> navicore_;
  std::vector<int> vhcalEnum_;
  std::vector<unsigned int> vDenseIdHcal_;
  PFHCALDenseIdNavigatorCore* pfHcalDenseIdNavigatorCore_;

private:
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

#endif  // RecoParticleFlow_PFClusterProducer_interface_PFHCALDenseIdNavigator_h
