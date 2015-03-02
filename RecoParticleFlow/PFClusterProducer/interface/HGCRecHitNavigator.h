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
    eeNav_ = new hgcee(iConfig.getParameter<edm::ParameterSet>("hgcee"));
    hefNav_ = new hgchef(iConfig.getParameter<edm::ParameterSet>("hgchef"));
    hebNav_ = new hgcheb(iConfig.getParameter<edm::ParameterSet>("hgcheb"));
  }

  void beginEvent(const edm::EventSetup& iSetup) {
    eeNav_->beginEvent(iSetup); 
    hefNav_->beginEvent(iSetup);
    hebNav_->beginEvent(iSetup);
  }

  void associateNeighbours(reco::PFRecHit& hit,std::auto_ptr<reco::PFRecHitCollection>& hits,edm::RefProd<reco::PFRecHitCollection>& refProd) {
    switch( hit.layer() ) {
    case D1:
      eeNav_->associateNeighbours(hit,hits,refProd);
      break;
    case D2:
      hefNav_->associateNeighbours(hit,hits,refProd);
      break;
    case D3:
      hebNav_->associateNeighbours(hit,hits,refProd);
      break;
    default:
      break;
    }     
  }

  virtual void associateNeighbours(reco::PFRecHit& hit,std::auto_ptr<reco::PFRecHitCollection>& hits,const DetIdToHitIdx& hitmap,edm::RefProd<reco::PFRecHitCollection>& refProd) override {
    switch( hit.layer() ) {
    case D1:
      eeNav_->associateNeighbours(hit,hits,hitmap,refProd);
      break;
    case D2:
      hefNav_->associateNeighbours(hit,hits,hitmap,refProd);
      break;
    case D3:
      hebNav_->associateNeighbours(hit,hits,hitmap,refProd);
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


