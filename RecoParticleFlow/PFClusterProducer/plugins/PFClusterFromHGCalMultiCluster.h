#ifndef __RecoParticleFlow_PFClusterProducer_PFClusterFromHGCalMultiCluster_H__
#define __RecoParticleFlow_PFClusterProducer_PFClusterFromHGCalMultiCluster_H__

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"

class PFClusterFromHGCalMultiCluster : public InitialClusteringStepBase {
public:
  PFClusterFromHGCalMultiCluster(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
      : InitialClusteringStepBase(conf, sumes) {
    filterByTracksterPID_ = conf.getParameter<bool>("filterByTracksterPID");
    pid_threshold_ = conf.getParameter<double>("pid_threshold");
    filter_on_categories_ = conf.getParameter<std::vector<int> >("filter_on_categories");

    clusterToken_ =
        sumes.consumes<std::vector<reco::HGCalMultiCluster> >(conf.getParameter<edm::InputTag>("clusterSrc"));
    tracksterToken_ = sumes.consumes<std::vector<ticl::Trackster> >(conf.getParameter<edm::InputTag>("tracksterSrc"));
  }

  ~PFClusterFromHGCalMultiCluster() override {}
  PFClusterFromHGCalMultiCluster(const PFClusterFromHGCalMultiCluster&) = delete;
  PFClusterFromHGCalMultiCluster& operator=(const PFClusterFromHGCalMultiCluster&) = delete;

  void updateEvent(const edm::Event&) final;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
                     const std::vector<bool>&,
                     const std::vector<bool>&,
                     reco::PFClusterCollection&) override;

private:
  bool filterByTracksterPID_;
  float pid_threshold_;
  std::vector<int> filter_on_categories_;

  edm::EDGetTokenT<std::vector<reco::HGCalMultiCluster> > clusterToken_;
  edm::Handle<std::vector<reco::HGCalMultiCluster> > clusterH_;

  edm::EDGetTokenT<std::vector<ticl::Trackster> > tracksterToken_;
  edm::Handle<std::vector<ticl::Trackster> > trackstersH_;
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory, PFClusterFromHGCalMultiCluster, "PFClusterFromHGCalMultiCluster");

#endif
