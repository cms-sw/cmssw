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
    _clusterToken = sumes.consumes<std::vector<reco::HGCalMultiCluster> >(
        conf.getParameter<edm::InputTag>("clusterSrc"));
  }
  virtual ~PFClusterFromHGCalMultiCluster() {}
  PFClusterFromHGCalMultiCluster(const PFClusterFromHGCalMultiCluster&) =
      delete;
  PFClusterFromHGCalMultiCluster& operator=(
      const PFClusterFromHGCalMultiCluster&) = delete;

  virtual void updateEvent(const edm::Event&) override final;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
                     const std::vector<bool>&, const std::vector<bool>&,
                     reco::PFClusterCollection&) override;

 private:
  edm::EDGetTokenT<std::vector<reco::HGCalMultiCluster> > _clusterToken;
  edm::Handle<std::vector<reco::HGCalMultiCluster> > _clusterH;
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory, PFClusterFromHGCalMultiCluster,
                  "PFClusterFromHGCalMultiCluster");

#endif
