#ifndef __LocalMaximumSeedFinder_H__
#define __LocalMaximumSeedFinder_H__

#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"

#include <unordered_map>
#include <tuple>

class LocalMaximumSeedFinder final : public SeedFinderBase {
 public:
  LocalMaximumSeedFinder(const edm::ParameterSet& conf);
  LocalMaximumSeedFinder(const LocalMaximumSeedFinder&) = delete;
  LocalMaximumSeedFinder& operator=(const LocalMaximumSeedFinder&) = delete;

  void findSeeds( const edm::Handle<reco::PFRecHitCollection>& input,
		  const std::vector<bool>& mask,
		  std::vector<bool>& seedable ) override;

 private:  
  const int _nNeighbours;

  const std::unordered_map<std::string,int> _layerMap;

  typedef std::tuple<std::vector<int> ,std::vector<double> , std::vector<double> > _i3tuple;

  std::array<_i3tuple, 35> _thresholds;
  static constexpr int layerOffset = 15;
};

DEFINE_EDM_PLUGIN(SeedFinderFactory,
		  LocalMaximumSeedFinder,"LocalMaximumSeedFinder");

#endif
