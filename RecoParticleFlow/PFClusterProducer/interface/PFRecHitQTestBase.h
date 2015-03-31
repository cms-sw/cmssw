#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitQTestBase_h
#define RecoParticleFlow_PFClusterProducer_PFRecHitQTestBase_h


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

class PFRecHitQTestBase {
 public:
  PFRecHitQTestBase() {}
  PFRecHitQTestBase(const edm::ParameterSet& iConfig) {}

  virtual void beginEvent(const edm::Event&,const edm::EventSetup&)=0;


  virtual bool test( reco::PFRecHit& ,const EcalRecHit&,bool&)=0;
  virtual bool test( reco::PFRecHit& ,const HBHERecHit&,bool&)=0;
  virtual bool test( reco::PFRecHit& ,const HFRecHit&,bool&)=0;
  virtual bool test( reco::PFRecHit& ,const HORecHit&,bool&)=0;
  virtual bool test( reco::PFRecHit& ,const CaloTower&,bool&)=0;
};
 

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<PFRecHitQTestBase*(const edm::ParameterSet&)> PFRecHitQTestFactory;
#endif
