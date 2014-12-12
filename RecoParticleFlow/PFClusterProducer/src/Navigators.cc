#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

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

class PFRecHitEcalBarrelNavigator : public PFRecHitCaloNavigator<EBDetId,EcalBarrelTopology> {
 public:
  PFRecHitEcalBarrelNavigator(const edm::ParameterSet& iConfig) {

  }

  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<CaloGeometryRecord>().get(geoHandle);
    topology_.reset( new EcalBarrelTopology(geoHandle) );
  }
};

class PFRecHitEcalEndcapNavigator : public PFRecHitCaloNavigator<EEDetId,EcalEndcapTopology> {
 public:
  PFRecHitEcalEndcapNavigator(const edm::ParameterSet& iConfig) {

  }

  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<CaloGeometryRecord>().get(geoHandle);
    topology_.reset( new EcalEndcapTopology(geoHandle) );
  }
};

class PFRecHitPreshowerNavigator : public PFRecHitCaloNavigator<ESDetId,EcalPreshowerTopology> {
 public:
  PFRecHitPreshowerNavigator(const edm::ParameterSet& iConfig) {

  }


  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<CaloGeometryRecord>().get(geoHandle);
    topology_.reset( new EcalPreshowerTopology(geoHandle) );
  }
};


class PFRecHitHCALNavigator : public PFRecHitCaloNavigator<HcalDetId,HcalTopology,false> {
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

class PFRecHitHCAL3DNavigator : public PFRecHitCaloNavigator<HcalDetId,HcalTopology,false,3> {
public:
  PFRecHitHCAL3DNavigator(const edm::ParameterSet& iConfig) {

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
    edm::ESHandle<HcalTopology> hcalTopology;
    iSetup.get<HcalRecNumberingRecord>().get( hcalTopology );
    topology_.reset( new CaloTowerTopology( hcalTopology.product() ) );    
  }
private:  
};

typedef PFRecHitDualNavigator<PFLayer::ECAL_BARREL,
			      PFRecHitEcalBarrelNavigator,
			      PFLayer::ECAL_ENDCAP,
			    PFRecHitEcalEndcapNavigator> PFRecHitECALNavigator;

typedef  PFRecHitDualNavigator<PFLayer::ECAL_BARREL,
			       PFRecHitEcalBarrelNavigatorWithTime,
			       PFLayer::ECAL_ENDCAP,
	   PFRecHitEcalEndcapNavigatorWithTime> PFRecHitECALNavigatorWithTime;


#include "Geometry/CaloTopology/interface/ShashlikTopology.h"
#include "Geometry/Records/interface/ShashlikNumberingRecord.h"
class PFRecHitShashlikNavigator : public PFRecHitCaloNavigator<EKDetId,ShashlikTopology,false>{
public:
  PFRecHitShashlikNavigator(const edm::ParameterSet& iConfig) { topology_ = NULL; }
  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<ShashlikTopology> topoHandle;
    iSetup.get<ShashlikNumberingRecord>().get(topoHandle);
    topology_.release();
    topology_.reset(topoHandle.product());
  }
};
class PFRecHitShashlikNavigatorWithTime : public PFRecHitCaloNavigatorWithTime<EKDetId,ShashlikTopology,false> {
 public:
  PFRecHitShashlikNavigatorWithTime(const edm::ParameterSet& iConfig):
    PFRecHitCaloNavigatorWithTime(iConfig)
    {

    }

  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<ShashlikTopology> topoHandle;
    iSetup.get<ShashlikNumberingRecord>().get(topoHandle);
    topology_.release();
    topology_.reset( topoHandle.product() );
  }
};
typedef PFRecHitDualNavigator<PFLayer::ECAL_BARREL,
			      PFRecHitEcalBarrelNavigator,
			      PFLayer::ECAL_ENDCAP,
			    PFRecHitShashlikNavigator> PFRecHitEBEKNavigator;

#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
class PFRecHitHGCEENavigator : public PFRecHitCaloNavigator<HGCEEDetId,HGCalTopology,false,3> {
  const std::string topoSource_;
public:
  PFRecHitHGCEENavigator(const edm::ParameterSet& iConfig) :
    topoSource_(iConfig.getParameter<std::string>("topologySource")) {
    topology_ = NULL;
  }
  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<HGCalTopology> topoHandle;
    iSetup.get<IdealGeometryRecord>().get(topoSource_,topoHandle);
    topology_.release();
    topology_.reset(topoHandle.product());
  }
};
class PFRecHitHGCHENavigator : public PFRecHitCaloNavigator<HGCHEDetId,HGCalTopology,false,3> {
  const std::string topoSource_;
public:
  PFRecHitHGCHENavigator(const edm::ParameterSet& iConfig) :
    topoSource_(iConfig.getParameter<std::string>("topologySource")) {
    topology_ = NULL;
  }
  void beginEvent(const edm::EventSetup& iSetup) {
    edm::ESHandle<HGCalTopology> topoHandle;
    iSetup.get<IdealGeometryRecord>().get(topoSource_,topoHandle);
    topology_.release();
    topology_.reset(topoHandle.product());
  }
};

typedef HGCRecHitNavigator<PFLayer::HGC_ECAL,
			   PFRecHitHGCEENavigator,
			   PFLayer::HGC_HCALF,
			   PFRecHitHGCHENavigator,
			   PFLayer::HGC_HCALB,
			   PFRecHitHGCHENavigator> PFRecHitHGCNavigator;

EDM_REGISTER_PLUGINFACTORY(PFRecHitNavigationFactory, "PFRecHitNavigationFactory");

DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalBarrelNavigator, "PFRecHitEcalBarrelNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalEndcapNavigator, "PFRecHitEcalEndcapNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitShashlikNavigator, "PFRecHitShashlikNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHGCEENavigator, "PFRecHitHGCEENavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHGCHENavigator, "PFRecHitHGCHENavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHGCNavigator, "PFRecHitHGCNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalBarrelNavigatorWithTime, "PFRecHitEcalBarrelNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalEndcapNavigatorWithTime, "PFRecHitEcalEndcapNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitShashlikNavigatorWithTime, "PFRecHitShashlikNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFECALHashNavigator, "PFECALHashNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitECALNavigator, "PFRecHitECALNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEBEKNavigator, "PFRecHitEBEKNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitECALNavigatorWithTime, "PFRecHitECALNavigatorWithTime");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitCaloTowerNavigator, "PFRecHitCaloTowerNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitPreshowerNavigator, "PFRecHitPreshowerNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHCALNavigator, "PFRecHitHCALNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHCAL3DNavigator, "PFRecHitHCAL3DNavigator");

