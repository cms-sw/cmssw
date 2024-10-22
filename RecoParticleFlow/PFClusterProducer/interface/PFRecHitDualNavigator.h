#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitDualNavigator_h
#define RecoParticleFlow_PFClusterProducer_PFRecHitDualNavigator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

template <PFLayer::Layer D1, typename barrel, PFLayer::Layer D2, typename endcap>
class PFRecHitDualNavigator : public PFRecHitNavigatorBase {
public:
  PFRecHitDualNavigator() = default;

  PFRecHitDualNavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc) {
    barrelNav_ = new barrel(iConfig.getParameter<edm::ParameterSet>("barrel"), cc);
    endcapNav_ = new endcap(iConfig.getParameter<edm::ParameterSet>("endcap"), cc);
  }

  void init(const edm::EventSetup& iSetup) override {
    barrelNav_->init(iSetup);
    endcapNav_->init(iSetup);
  }

  void associateNeighbours(reco::PFRecHit& hit,
                           std::unique_ptr<reco::PFRecHitCollection>& hits,
                           edm::RefProd<reco::PFRecHitCollection>& refProd) override {
    if (hit.layer() == D1)
      barrelNav_->associateNeighbours(hit, hits, refProd);
    else if (hit.layer() == D2)
      endcapNav_->associateNeighbours(hit, hits, refProd);
  }

protected:
  barrel* barrelNav_;
  endcap* endcapNav_;
};

#endif
