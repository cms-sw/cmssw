#ifndef RecoParticleFlow_PFClusterProducer_HGCRecHitNavigator_h
#define RecoParticleFlow_PFClusterProducer_HGCRecHitNavigator_h


#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

template <ForwardSubdetector D1, typename hgcee,
          ForwardSubdetector D2, typename hgchef,
          ForwardSubdetector D3, typename hgcheb>
class HGCRecHitNavigator : public PFRecHitNavigatorBase {
 public:
  HGCRecHitNavigator() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    
    desc.add<std::string>("name","PFRecHitHGCNavigator");

    edm::ParameterSetDescription descee;
    descee.add<std::string>("name","PFRecHitHGCEENavigator");
    descee.add<std::string>("topologySource","HGCalEESensitive");
    desc.add<edm::ParameterSetDescription>("hgcee", descee);
    
    edm::ParameterSetDescription deschef;
    deschef.add<std::string>("name","PFRecHitHGCHENavigator");
    deschef.add<std::string>("topologySource","HGCalHESiliconSensitive");
    desc.add<edm::ParameterSetDescription>("hgchef", deschef);
    
    edm::ParameterSetDescription descheb;
    deschef.add<std::string>("name","PFRecHitHGCHENavigator");
    deschef.add<std::string>("topologySource","HGCalHEScintillatorSensitive");
    desc.add<edm::ParameterSetDescription>("hgcheb", descheb);
    
    descriptions.add("navigator", desc);
  }
  

  HGCRecHitNavigator(const edm::ParameterSet& iConfig) {
    if( iConfig.exists("hgcee") ) {
      eeNav_ = new hgcee(iConfig.getParameter<edm::ParameterSet>("hgcee"));
    } else {
      eeNav_ = nullptr;
    }
    if( iConfig.exists("hgchef") ) {
      hefNav_ = new hgchef(iConfig.getParameter<edm::ParameterSet>("hgchef"));
    } else {
      hefNav_ = nullptr;
    }
    if( iConfig.exists("hgcheb") ) {
      hebNav_ = new hgcheb(iConfig.getParameter<edm::ParameterSet>("hgcheb"));
    } else {
      hebNav_ = nullptr;
    }
  }

  void beginEvent(const edm::EventSetup& iSetup) override {
    if( nullptr != eeNav_ )  eeNav_->beginEvent(iSetup); 
    if( nullptr != hefNav_ ) hefNav_->beginEvent(iSetup);
    if( nullptr != hebNav_ ) hebNav_->beginEvent(iSetup);
  }

  void associateNeighbours(reco::PFRecHit& hit,std::unique_ptr<reco::PFRecHitCollection>& hits,edm::RefProd<reco::PFRecHitCollection>& refProd) override {
    switch( DetId(hit.detId()).subdetId() ) {
    case D1:
      if( nullptr != eeNav_ )  eeNav_->associateNeighbours(hit,hits,refProd);
      break;
    case D2:
      if( nullptr != hefNav_ ) hefNav_->associateNeighbours(hit,hits,refProd);
      break;
    case D3:
      if( nullptr != hebNav_ ) hebNav_->associateNeighbours(hit,hits,refProd);
      break;
    default:
      break;
    }     
  }
  
 protected:
      hgcee  *eeNav_;
      hgchef *hefNav_;
      hgcheb *hebNav_;
};

#endif
