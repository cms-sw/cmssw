#ifndef __LocalMaximum2DSeedFinder_H__
#define __LocalMaximum2DSeedFinder_H__

#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"

#include <unordered_map>

class LocalMaximum2DSeedFinder : public SeedFinderBase {
 public:
  LocalMaximum2DSeedFinder(const edm::ParameterSet& conf);
  LocalMaximum2DSeedFinder(const LocalMaximum2DSeedFinder&) = delete;
  LocalMaximum2DSeedFinder& operator=(const LocalMaximum2DSeedFinder&) = delete;

  void findSeeds( const edm::Handle<reco::PFRecHitCollection>& input,
		  const std::vector<bool>& mask,
		  std::vector<bool>& seedable );

 private:  
  const unsigned _nNeighbours;

  static const reco::PFRecHitRefVector _noNeighbours;
  const std::unordered_map<std::string,int> _layerMap;
  std::unordered_map<int,std::pair<double,double> > 
    _thresholds;
};

DEFINE_EDM_PLUGIN(SeedFinderFactory,
		  LocalMaximum2DSeedFinder,"LocalMaximum2DSeedFinder");

#endif
