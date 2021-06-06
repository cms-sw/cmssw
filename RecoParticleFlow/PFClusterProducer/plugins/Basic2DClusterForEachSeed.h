#ifndef __Basic2DClusterForEachSeed_H__
#define __Basic2DClusterForEachSeed_H__

#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

class Basic2DClusterForEachSeed : public InitialClusteringStepBase {
public:
  Basic2DClusterForEachSeed(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
      : InitialClusteringStepBase(conf, cc) {}
  ~Basic2DClusterForEachSeed() override = default;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
                     const std::vector<bool>&,
                     const std::vector<bool>&,
                     reco::PFClusterCollection&) override;
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory, Basic2DClusterForEachSeed, "Basic2DClusterForEachSeed");

#endif
