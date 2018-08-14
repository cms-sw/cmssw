#ifndef __RecoParticleFlow_PFClusterProducer_PassThruSeedFinder_H__
#define __RecoParticleFlow_PFClusterProducer_PassThruSeedFinder_H__

#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"

#include <unordered_map>

class PassThruSeedFinder : public SeedFinderBase {
 public:
  PassThruSeedFinder(const edm::ParameterSet& conf);
  PassThruSeedFinder(const PassThruSeedFinder&) = delete;
  PassThruSeedFinder& operator=(const PassThruSeedFinder&) = delete;

  void findSeeds( const edm::Handle<reco::PFRecHitCollection>& input,
		  const std::vector<bool>& mask,
		  std::vector<bool>& seedable ) override;

 private:  
};

DEFINE_EDM_PLUGIN(SeedFinderFactory,
		  PassThruSeedFinder,"PassThruSeedFinder");

#endif
