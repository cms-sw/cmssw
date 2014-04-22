#ifndef __LocalMaximumSeedFinder_H__
#define __LocalMaximumSeedFinder_H__

#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"

#include <unordered_map>

class LocalMaximumSeedFinder : public SeedFinderBase {
 public:
  LocalMaximumSeedFinder(const edm::ParameterSet& conf);
  LocalMaximumSeedFinder(const LocalMaximumSeedFinder&) = delete;
  LocalMaximumSeedFinder& operator=(const LocalMaximumSeedFinder&) = delete;

  void findSeeds( const edm::Handle<reco::PFRecHitCollection>& input,
		  const std::vector<bool>& mask,
		  std::vector<bool>& seedable );

 private:  
  const int _nNeighbours;

  static const reco::PFRecHitRefVector _noNeighbours;
  const std::unordered_map<std::string,int> _layerMap;
  std::unordered_map<int,std::pair<double,double> > 
    _thresholds;
};

DEFINE_EDM_PLUGIN(SeedFinderFactory,
		  LocalMaximumSeedFinder,"LocalMaximumSeedFinder");

#endif
