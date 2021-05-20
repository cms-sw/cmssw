#ifndef __RecoParticleFlow_PFClusterProducer_PFClusterFromHGCalTrackster_H__
#define __RecoParticleFlow_PFClusterProducer_PFClusterFromHGCalTrackster_H__

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"

class PFClusterFromHGCalTrackster : public InitialClusteringStepBase {
public:
  PFClusterFromHGCalTrackster(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
      : InitialClusteringStepBase(conf, cc) {
    filterByTracksterPID_ = conf.getParameter<bool>("filterByTracksterPID");
    pid_threshold_ = conf.getParameter<double>("pid_threshold");
    filter_on_categories_ = conf.getParameter<std::vector<int> >("filter_on_categories");

    tracksterToken_ = cc.consumes<std::vector<ticl::Trackster> >(conf.getParameter<edm::InputTag>("tracksterSrc"));
    clusterToken_ = cc.consumes<reco::CaloClusterCollection>(conf.getParameter<edm::InputTag>("clusterSrc"));
  }

  ~PFClusterFromHGCalTrackster() override {}
  PFClusterFromHGCalTrackster(const PFClusterFromHGCalTrackster&) = delete;
  PFClusterFromHGCalTrackster& operator=(const PFClusterFromHGCalTrackster&) = delete;

  void updateEvent(const edm::Event&) final;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
                     const std::vector<bool>&,
                     const std::vector<bool>&,
                     reco::PFClusterCollection&) override;

private:
  bool filterByTracksterPID_;
  float pid_threshold_;
  std::vector<int> filter_on_categories_;

  edm::EDGetTokenT<std::vector<ticl::Trackster> > tracksterToken_;
  edm::Handle<std::vector<ticl::Trackster> > trackstersH_;

  edm::EDGetTokenT<reco::CaloClusterCollection> clusterToken_;
  edm::Handle<reco::CaloClusterCollection> clusterH_;
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory, PFClusterFromHGCalTrackster, "PFClusterFromHGCalTrackster");

#endif
