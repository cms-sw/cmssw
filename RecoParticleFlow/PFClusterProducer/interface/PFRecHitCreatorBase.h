#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitCreatorBase_h
#define RecoParticleFlow_PFClusterProducer_PFRecHitCreatorBase_h



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

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitQTestBase.h"
#include <memory>


class PFRecHitCreatorBase {
 public:
  PFRecHitCreatorBase() {}
  PFRecHitCreatorBase(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC) {
    std::vector<edm::ParameterSet> qTests =   iConfig.getParameter<std::vector<edm::ParameterSet> >("qualityTests");
    for (unsigned int i=0;i<qTests.size();++i) {
      std::string name = qTests.at(i).getParameter<std::string>("name");
      qualityTests_.emplace_back(PFRecHitQTestFactory::get()->create(name,qTests.at(i)));
    }
  }



  virtual void importRecHits(std::auto_ptr<reco::PFRecHitCollection>&,std::auto_ptr<reco::PFRecHitCollection>& ,const edm::Event&,const edm::EventSetup&)=0;

 protected:
  std::vector<std::unique_ptr<PFRecHitQTestBase> > qualityTests_;

};
 

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<PFRecHitCreatorBase*(const edm::ParameterSet&,edm::ConsumesCollector&)> PFRecHitFactory;
#endif
