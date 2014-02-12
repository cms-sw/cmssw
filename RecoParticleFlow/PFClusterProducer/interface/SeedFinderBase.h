#ifndef __SeedFinderBase_H__
#define __SeedFinderBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

class SeedFinderBase {
 public:
  SeedFinderBase(const edm::ParameterSet& conf):
    _seedingThreshold(conf.getParameter<float>("seedingThreshold")),
    _algoName(conf.getParameter<std::string>("algoName")) { }
  SeedFinderBase(const SeedFinderBase&) = delete;
  SeedFinderBase& operator=(const SeedFinderBase&) = delete;

  virtual void findSeeds(const reco::PFRecHitRefVector& input, 
			 const std::vector<bool>& mask,
			 std::vector<bool>& seedable) = 0;

  void reset() { _seeds.clear(); }

  const std::string& name() const { return _algoName; }

 protected:
  const float _seedingThreshold;
  reco::PFRecHitRefVector _seeds;

 private:
  const std::string _algoName;
  
};

#endif
