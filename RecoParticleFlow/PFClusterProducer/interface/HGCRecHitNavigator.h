#ifndef RecoParticleFlow_PFClusterProducer_HGCRecHitNavigator_h
#define RecoParticleFlow_PFClusterProducer_HGCRecHitNavigator_h


#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"



template <PFLayer::Layer D1, typename hgcee,
          PFLayer::Layer D2, typename hgchef,
          PFLayer::Layer D3, typename hgcheb>
class HGCRecHitNavigator : public PFRecHitNavigatorBase {
 public:
  HGCRecHitNavigator() {
  }



  HGCRecHitNavigator(const edm::ParameterSet& iConfig){
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

  void beginEvent(const edm::EventSetup& iSetup) {
    if( nullptr != eeNav_ )  eeNav_->beginEvent(iSetup); 
    if( nullptr != hefNav_ ) hefNav_->beginEvent(iSetup);
    if( nullptr != hebNav_ ) hebNav_->beginEvent(iSetup);
  }

  void associateNeighbours(reco::PFRecHit& hit,std::auto_ptr<reco::PFRecHitCollection>& hits,edm::RefProd<reco::PFRecHitCollection>& refProd) {
    switch( hit.layer() ) {
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

  virtual void associateNeighbours(reco::PFRecHit& hit,std::auto_ptr<reco::PFRecHitCollection>& hits,const DetIdToHitIdx& hitmap,edm::RefProd<reco::PFRecHitCollection>& refProd) override {
    switch( hit.layer() ) {
    case D1:
      if( nullptr != eeNav_ )  eeNav_->associateNeighbours(hit,hits,hitmap,refProd);
      break;
    case D2:
      if( nullptr != hefNav_ ) hefNav_->associateNeighbours(hit,hits,hitmap,refProd);
      break;
    case D3:
      if( nullptr != hebNav_ ) hebNav_->associateNeighbours(hit,hits,hitmap,refProd);
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


