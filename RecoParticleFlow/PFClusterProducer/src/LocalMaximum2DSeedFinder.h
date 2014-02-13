#ifndef __LocalMaximum2DSeedFinder_H__
#define __LocalMaximum2DSeedFinder_H__

#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"

class LocalMaximum2DSeedFinder : public SeedFinderBase {
 public:
  LocalMaximum2DSeedFinder(const edm::ParameterSet& conf) :
    SeedFinderBase(conf),
    _seedingThresholdPt2(std::pow(conf.getParameter<double>("seedingThresholdPt"),2.0)),
    _nNeighbours(conf.getParameter<unsigned>("nNeighbours")) { }
  LocalMaximum2DSeedFinder(const LocalMaximum2DSeedFinder&) = delete;
  LocalMaximum2DSeedFinder& operator=(const LocalMaximum2DSeedFinder&) = delete;

  void findSeeds( const edm::Handle<reco::PFRecHitCollection>& input,
		  const std::vector<bool>& mask,
		  std::vector<bool>& seedable );

 private:
  const float _seedingThresholdPt2;
  const unsigned _nNeighbours;

  static const std::vector<unsigned> _noNeighbours;
};

#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderFactory.h"
DEFINE_EDM_PLUGIN(SeedFinderFactory,
		  LocalMaximum2DSeedFinder,"LocalMaximum2DSeedFinder");

#endif
