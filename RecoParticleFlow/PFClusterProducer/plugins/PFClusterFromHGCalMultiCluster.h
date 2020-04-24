#ifndef __RecoParticleFlow_PFClusterProducer_PFClusterFromHGCalMultiCluster_H__
#define __RecoParticleFlow_PFClusterProducer_PFClusterFromHGCalMultiCluster_H__

#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"

class PFClusterFromHGCalMultiCluster : public InitialClusteringStepBase {
 public:
  PFClusterFromHGCalMultiCluster(const edm::ParameterSet& conf,
                                 edm::ConsumesCollector& sumes)
      : InitialClusteringStepBase(conf, sumes) {
    clusterToken_ = sumes.consumes<std::vector<reco::HGCalMultiCluster> >(
        conf.getParameter<edm::InputTag>("clusterSrc"));
  }
  ~PFClusterFromHGCalMultiCluster() override {}
  PFClusterFromHGCalMultiCluster(const PFClusterFromHGCalMultiCluster&) =
      delete;
  PFClusterFromHGCalMultiCluster& operator=(
      const PFClusterFromHGCalMultiCluster&) = delete;

  void updateEvent(const edm::Event&) final;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
                     const std::vector<bool>&, const std::vector<bool>&,
                     reco::PFClusterCollection&) override;

 private:
  edm::EDGetTokenT<std::vector<reco::HGCalMultiCluster> > clusterToken_;
  edm::Handle<std::vector<reco::HGCalMultiCluster> > clusterH_;
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory, PFClusterFromHGCalMultiCluster,
                  "PFClusterFromHGCalMultiCluster");

#endif
