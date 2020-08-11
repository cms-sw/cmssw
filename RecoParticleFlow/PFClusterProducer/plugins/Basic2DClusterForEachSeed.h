#ifndef __Basic2DClusterForEachSeed_H__
#define __Basic2DClusterForEachSeed_H__

#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "CommonTools/Utils/interface/DynArray.h"

class Basic2DClusterForEachSeed : public InitialClusteringStepBase {
  typedef Basic2DClusterForEachSeed B2DGT;

public:
  Basic2DClusterForEachSeed(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
      : InitialClusteringStepBase(conf, sumes) {}
  ~Basic2DClusterForEachSeed() override = default;
  Basic2DClusterForEachSeed(const B2DGT&) = delete;
  B2DGT& operator=(const B2DGT&) = delete;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
                     const std::vector<bool>&,
                     const std::vector<bool>&,
                     reco::PFClusterCollection&) override;
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory, Basic2DClusterForEachSeed, "Basic2DClusterForEachSeed");

#endif
