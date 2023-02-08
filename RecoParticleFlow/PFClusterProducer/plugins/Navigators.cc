#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitFakeNavigator.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitDualNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCaloNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCaloNavigatorWithTime.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFECALHashNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/HGCRecHitNavigator.h"

class PFRecHitEcalBarrelNavigatorWithTime : public PFRecHitCaloNavigatorWithTime<EBDetId, EcalBarrelTopology> {
public:
  PFRecHitEcalBarrelNavigatorWithTime(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : PFRecHitCaloNavigatorWithTime(iConfig, cc), geomToken_(cc.esConsumes<edm::Transition::BeginRun>()) {}

  void init(const edm::EventSetup& iSetup) override {
    edm::ESHandle<CaloGeometry> geoHandle = iSetup.getHandle(geomToken_);
    topology_ = std::make_unique<EcalBarrelTopology>(*geoHandle);
  }

private:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

class PFRecHitEcalEndcapNavigatorWithTime : public PFRecHitCaloNavigatorWithTime<EEDetId, EcalEndcapTopology> {
public:
  PFRecHitEcalEndcapNavigatorWithTime(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : PFRecHitCaloNavigatorWithTime(iConfig, cc), geomToken_(cc.esConsumes<edm::Transition::BeginRun>()) {}

  void init(const edm::EventSetup& iSetup) override {
    edm::ESHandle<CaloGeometry> geoHandle = iSetup.getHandle(geomToken_);
    topology_ = std::make_unique<EcalEndcapTopology>(*geoHandle);
  }

private:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

class PFRecHitEcalBarrelNavigator final : public PFRecHitCaloNavigator<EBDetId, EcalBarrelTopology> {
public:
  PFRecHitEcalBarrelNavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : geomToken_(cc.esConsumes<edm::Transition::BeginRun>()) {}

  void init(const edm::EventSetup& iSetup) override {
    edm::ESHandle<CaloGeometry> geoHandle = iSetup.getHandle(geomToken_);
    topology_ = std::make_unique<EcalBarrelTopology>(*geoHandle);
  }

private:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

class PFRecHitEcalEndcapNavigator final : public PFRecHitCaloNavigator<EEDetId, EcalEndcapTopology> {
public:
  PFRecHitEcalEndcapNavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : geomToken_(cc.esConsumes<edm::Transition::BeginRun>()) {}

  void init(const edm::EventSetup& iSetup) override {
    edm::ESHandle<CaloGeometry> geoHandle = iSetup.getHandle(geomToken_);
    topology_ = std::make_unique<EcalEndcapTopology>(*geoHandle);
  }

private:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

class PFRecHitPreshowerNavigator final : public PFRecHitCaloNavigator<ESDetId, EcalPreshowerTopology> {
public:
  PFRecHitPreshowerNavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc) {}

  void init(const edm::EventSetup& iSetup) override { topology_ = std::make_unique<EcalPreshowerTopology>(); }
};

class PFRecHitHCALDenseIdNavigator final : public PFHCALDenseIdNavigator<HcalDetId, HcalTopology, false> {
public:
  PFRecHitHCALDenseIdNavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : PFHCALDenseIdNavigator(iConfig, cc) {}
};

class PFRecHitHCALNavigator : public PFRecHitCaloNavigator<HcalDetId, HcalTopology, false> {
public:
  PFRecHitHCALNavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : hcalToken_(cc.esConsumes<edm::Transition::BeginRun>()) {}

  void init(const edm::EventSetup& iSetup) override {
    edm::ESHandle<HcalTopology> hcalTopology = iSetup.getHandle(hcalToken_);
    topology_.release();
    topology_.reset(hcalTopology.product());
  }

private:
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;
};

class PFRecHitHCALNavigatorWithTime : public PFRecHitCaloNavigatorWithTime<HcalDetId, HcalTopology, false> {
public:
  PFRecHitHCALNavigatorWithTime(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : PFRecHitCaloNavigatorWithTime(iConfig, cc), hcalToken_(cc.esConsumes<edm::Transition::BeginRun>()) {}

  void init(const edm::EventSetup& iSetup) override {
    edm::ESHandle<HcalTopology> hcalTopology = iSetup.getHandle(hcalToken_);
    topology_.release();
    topology_.reset(hcalTopology.product());
  }

private:
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;
};

class PFRecHitCaloTowerNavigator : public PFRecHitCaloNavigator<CaloTowerDetId, CaloTowerTopology> {
public:
  PFRecHitCaloTowerNavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : caloToken_(cc.esConsumes<edm::Transition::BeginRun>()) {}

  void init(const edm::EventSetup& iSetup) override {
    edm::ESHandle<CaloTowerTopology> caloTowerTopology = iSetup.getHandle(caloToken_);
    topology_.release();
    topology_.reset(caloTowerTopology.product());
  }

private:
  edm::ESGetToken<CaloTowerTopology, HcalRecNumberingRecord> caloToken_;
};

typedef PFRecHitDualNavigator<PFLayer::ECAL_BARREL,
                              PFRecHitEcalBarrelNavigator,
                              PFLayer::ECAL_ENDCAP,
                              PFRecHitEcalEndcapNavigator>
    PFRecHitECALNavigator;

typedef PFRecHitDualNavigator<PFLayer::ECAL_BARREL,
                              PFRecHitEcalBarrelNavigatorWithTime,
                              PFLayer::ECAL_ENDCAP,
                              PFRecHitEcalEndcapNavigatorWithTime>
    PFRecHitECALNavigatorWithTime;

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

class PFRecHitHGCEENavigator : public PFRecHitFakeNavigator<HGCEEDetId> {
public:
  PFRecHitHGCEENavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc) {}

  void init(const edm::EventSetup& iSetup) override {}
};

class PFRecHitHGCHENavigator : public PFRecHitFakeNavigator<HGCHEDetId> {
public:
  PFRecHitHGCHENavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc) {}

  void init(const edm::EventSetup& iSetup) override {}
};

class PFRecHitHGCHexNavigator : public PFRecHitFakeNavigator<HGCalDetId> {
public:
  PFRecHitHGCHexNavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc) {}

  void init(const edm::EventSetup& iSetup) override {}
};

typedef HGCRecHitNavigator<HGCEE, PFRecHitHGCHexNavigator, HGCHEF, PFRecHitHGCHexNavigator, HGCHEB, PFRecHitHGCHENavigator>
    PFRecHitHGCNavigator;

EDM_REGISTER_PLUGINFACTORY(PFRecHitNavigationFactory, "PFRecHitNavigationFactory");

DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalBarrelNavigator, "PFRecHitEcalBarrelNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalEndcapNavigator, "PFRecHitEcalEndcapNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory,
                  PFRecHitEcalBarrelNavigatorWithTime,
                  "PFRecHitEcalBarrelNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory,
                  PFRecHitEcalEndcapNavigatorWithTime,
                  "PFRecHitEcalEndcapNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFECALHashNavigator, "PFECALHashNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitECALNavigator, "PFRecHitECALNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitECALNavigatorWithTime, "PFRecHitECALNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitCaloTowerNavigator, "PFRecHitCaloTowerNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitPreshowerNavigator, "PFRecHitPreshowerNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHCALDenseIdNavigator, "PFRecHitHCALDenseIdNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHCALNavigator, "PFRecHitHCALNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHCALNavigatorWithTime, "PFRecHitHCALNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHGCEENavigator, "PFRecHitHGCEENavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHGCHENavigator, "PFRecHitHGCHENavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHGCNavigator, "PFRecHitHGCNavigator");
