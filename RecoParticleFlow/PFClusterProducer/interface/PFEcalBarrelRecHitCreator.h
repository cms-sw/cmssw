#ifndef RecoParticleFlow_PFClusterProducer_PFEcalBarrelRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFEcalBarrelRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

class PFEcalBarrelRecHitCreator : public PFRecHitCreatorBase {
public:
  PFEcalBarrelRecHitCreator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : PFRecHitCreatorBase(iConfig, cc),
        recHitToken_(cc.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("src"))),
        triggerTowerMap_(nullptr),
        geomToken_(cc.esConsumes()),
        towerToken_(cc.esConsumes<edm::Transition::BeginRun>()) {
    auto srF = iConfig.getParameter<edm::InputTag>("srFlags");
    if (not srF.label().empty())
      srFlagToken_ = cc.consumes<EBSrFlagCollection>(srF);
  }

  void importRecHits(std::unique_ptr<reco::PFRecHitCollection>& out,
                     std::unique_ptr<reco::PFRecHitCollection>& cleaned,
                     const edm::Event& iEvent,
                     const edm::EventSetup& iSetup) override {
    beginEvent(iEvent, iSetup);

    edm::Handle<EcalRecHitCollection> recHitHandle;

    edm::ESHandle<CaloGeometry> geoHandle = iSetup.getHandle(geomToken_);

    bool useSrF = false;
    if (not srFlagToken_.isUninitialized()) {
      iEvent.getByToken(srFlagToken_, srFlagHandle_);
      useSrF = true;
    }

    // get the ecal geometry
    const CaloSubdetectorGeometry* gTmp = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);

    const EcalBarrelGeometry* ecalGeo = dynamic_cast<const EcalBarrelGeometry*>(gTmp);

    iEvent.getByToken(recHitToken_, recHitHandle);
    for (const auto& erh : *recHitHandle) {
      const DetId& detid = erh.detid();
      auto energy = erh.energy();
      auto time = erh.time();
      auto flags = erh.flagsBits();
      bool hi = (useSrF ? isHighInterest(detid) : true);

      const auto thisCell = ecalGeo->getGeometry(detid);

      // find rechit geometry
      if (!thisCell) {
        throw cms::Exception("PFEcalBarrelRecHitCreator") << "detid " << detid.rawId() << "not found in geometry";
      }

      out->emplace_back(thisCell, detid.rawId(), PFLayer::ECAL_BARREL, energy, flags);

      auto& rh = out->back();

      bool rcleaned = false;
      bool keep = true;

      //Apply Q tests
      for (const auto& qtest : qualityTests_) {
        if (!qtest->test(rh, erh, rcleaned, hi)) {
          keep = false;
        }
      }

      if (keep) {
        rh.setTime(time);
        rh.setDepth(1);
      } else {
        if (rcleaned)
          cleaned->push_back(std::move(out->back()));
        out->pop_back();
      }
    }
  }

  void init(const edm::EventSetup& es) override { triggerTowerMap_ = &es.getData(towerToken_); }

protected:
  bool isHighInterest(const EBDetId& detid) {
    bool result = false;
    auto srf = srFlagHandle_->find(readOutUnitOf(detid));
    if (srf == srFlagHandle_->end())
      return false;
    else
      result = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK) == EcalSrFlag::SRF_FULL);
    return result;
  }

  EcalTrigTowerDetId readOutUnitOf(const EBDetId& detid) const { return triggerTowerMap_->towerOf(detid); }

  edm::EDGetTokenT<EcalRecHitCollection> recHitToken_;
  edm::EDGetTokenT<EBSrFlagCollection> srFlagToken_;

  // ECAL trigger tower mapping
  const EcalTrigTowerConstituentsMap* triggerTowerMap_;
  // selective readout flags collection
  edm::Handle<EBSrFlagCollection> srFlagHandle_;

private:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> towerToken_;
};

#endif
