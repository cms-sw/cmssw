#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitDualNavigator_h
#define RecoParticleFlow_PFClusterProducer_PFRecHitDualNavigator_h


#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"



template <PFLayer::Layer D1, typename barrel,PFLayer::Layer D2, typename endcap>
class PFRecHitDualNavigator : public PFRecHitNavigatorBase {
 public:
  PFRecHitDualNavigator() = default;



  PFRecHitDualNavigator(const edm::ParameterSet& iConfig){
    barrelNav_ = new barrel(iConfig.getParameter<edm::ParameterSet>("barrel"));
    endcapNav_ = new endcap(iConfig.getParameter<edm::ParameterSet>("endcap"));
  }

  void beginEvent(const edm::EventSetup& iSetup) override {
    barrelNav_->beginEvent(iSetup); 
    endcapNav_->beginEvent(iSetup);

  }

  void associateNeighbours(reco::PFRecHit& hit,std::unique_ptr<reco::PFRecHitCollection>& hits,edm::RefProd<reco::PFRecHitCollection>& refProd) override {
      if (hit.layer() ==  D1)
	barrelNav_->associateNeighbours(hit,hits,refProd);
      else if (hit.layer() ==  D2)
	endcapNav_->associateNeighbours(hit,hits,refProd);
  }

 protected:
      barrel *barrelNav_;
      endcap *endcapNav_;


};

#endif


