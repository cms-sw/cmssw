#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitFakeNavigator.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitDualNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCaloNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCaloNavigatorWithTime.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFECALHashNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/HGCRecHitNavigator.h"

class PFRecHitEcalBarrelNavigatorWithTime : public PFRecHitCaloNavigatorWithTime<EBDetId,EcalBarrelTopology> {
 public:
  PFRecHitEcalBarrelNavigatorWithTime(const edm::ParameterSet& iConfig):
    PFRecHitCaloNavigatorWithTime(iConfig)
    {

    }

  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<CaloGeometryRecord>().get(geoHandle);
    topology_.reset( new EcalBarrelTopology(geoHandle) );
  }
};

class PFRecHitEcalEndcapNavigatorWithTime : public PFRecHitCaloNavigatorWithTime<EEDetId,EcalEndcapTopology> {
 public:
  PFRecHitEcalEndcapNavigatorWithTime(const edm::ParameterSet& iConfig):
    PFRecHitCaloNavigatorWithTime(iConfig)
    {

    }

  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<CaloGeometryRecord>().get(geoHandle);
    topology_.reset( new EcalEndcapTopology(geoHandle) );
  }
};

class PFRecHitEcalBarrelNavigator final : public PFRecHitCaloNavigator<EBDetId,EcalBarrelTopology> {
 public:
  PFRecHitEcalBarrelNavigator(const edm::ParameterSet& iConfig) {

  }

  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<CaloGeometryRecord>().get(geoHandle);
    topology_.reset( new EcalBarrelTopology(geoHandle) );
  }
};

class PFRecHitEcalEndcapNavigator final : public PFRecHitCaloNavigator<EEDetId,EcalEndcapTopology> {
 public:
  PFRecHitEcalEndcapNavigator(const edm::ParameterSet& iConfig) {

  }

  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<CaloGeometryRecord>().get(geoHandle);
    topology_.reset( new EcalEndcapTopology(geoHandle) );
  }
};

class PFRecHitPreshowerNavigator final : public PFRecHitCaloNavigator<ESDetId,EcalPreshowerTopology> {
 public:
  PFRecHitPreshowerNavigator(const edm::ParameterSet& iConfig) {

  }


  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<CaloGeometryRecord>().get(geoHandle);
    topology_.reset( new EcalPreshowerTopology(geoHandle) );
  }
};


class PFRecHitHCALNavigator final : public PFRecHitCaloNavigator<HcalDetId,HcalTopology,false> {
 public:
  PFRecHitHCALNavigator(const edm::ParameterSet& iConfig) {

  }


  void beginEvent(const edm::EventSetup& iSetup) {    
      edm::ESHandle<HcalTopology> hcalTopology;
      iSetup.get<HcalRecNumberingRecord>().get( hcalTopology );
      topology_.release();
      topology_.reset(hcalTopology.product());
  }
};
class PFRecHitHCALNavigatorWithTime : public PFRecHitCaloNavigatorWithTime<HcalDetId,HcalTopology,false> {
 public:
  PFRecHitHCALNavigatorWithTime(const edm::ParameterSet& iConfig):
    PFRecHitCaloNavigatorWithTime(iConfig)
  {
    
  }


  void beginEvent(const edm::EventSetup& iSetup) {    
      edm::ESHandle<HcalTopology> hcalTopology;
      iSetup.get<HcalRecNumberingRecord>().get( hcalTopology );
      topology_.release();
      topology_.reset(hcalTopology.product());
  }
};


class PFRecHitCaloTowerNavigator : public PFRecHitCaloNavigator<CaloTowerDetId,CaloTowerTopology> {
 public:
  PFRecHitCaloTowerNavigator(const edm::ParameterSet& iConfig) {

  }


  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<CaloTowerTopology> caloTowerTopology;
    iSetup.get<HcalRecNumberingRecord>().get(caloTowerTopology);
    topology_.release();
    topology_.reset(caloTowerTopology.product());
  }
};

typedef PFRecHitDualNavigator<PFLayer::ECAL_BARREL,
			      PFRecHitEcalBarrelNavigator,
			      PFLayer::ECAL_ENDCAP,
			    PFRecHitEcalEndcapNavigator> PFRecHitECALNavigator;

typedef  PFRecHitDualNavigator<PFLayer::ECAL_BARREL,
			       PFRecHitEcalBarrelNavigatorWithTime,
			       PFLayer::ECAL_ENDCAP,
	   PFRecHitEcalEndcapNavigatorWithTime> PFRecHitECALNavigatorWithTime;

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"

class PFRecHitHGCEENavigator : public PFRecHitFakeNavigator<HGCEEDetId> {
public:
  PFRecHitHGCEENavigator(const edm::ParameterSet& iConfig) {
  }

  void beginEvent(const edm::EventSetup& iSetup) {      
  }
};

class PFRecHitHGCHENavigator : public PFRecHitFakeNavigator<HGCHEDetId> {
public:
  PFRecHitHGCHENavigator(const edm::ParameterSet& iConfig) {
  }

  void beginEvent(const edm::EventSetup& iSetup) {      
  }
};

typedef HGCRecHitNavigator<HGCEE,
			   PFRecHitHGCEENavigator,
			   HGCHEF,
			   PFRecHitHGCHENavigator,
			   HGCHEB,
			   PFRecHitHGCHENavigator> PFRecHitHGCNavigator;

EDM_REGISTER_PLUGINFACTORY(PFRecHitNavigationFactory, "PFRecHitNavigationFactory");

DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalBarrelNavigator, "PFRecHitEcalBarrelNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalEndcapNavigator, "PFRecHitEcalEndcapNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalBarrelNavigatorWithTime, "PFRecHitEcalBarrelNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalEndcapNavigatorWithTime, "PFRecHitEcalEndcapNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFECALHashNavigator, "PFECALHashNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitECALNavigator, "PFRecHitECALNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitECALNavigatorWithTime, "PFRecHitECALNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitCaloTowerNavigator, "PFRecHitCaloTowerNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitPreshowerNavigator, "PFRecHitPreshowerNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHCALNavigator, "PFRecHitHCALNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHCALNavigatorWithTime, "PFRecHitHCALNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHGCEENavigator, "PFRecHitHGCEENavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHGCHENavigator, "PFRecHitHGCHENavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHGCNavigator, "PFRecHitHGCNavigator");

