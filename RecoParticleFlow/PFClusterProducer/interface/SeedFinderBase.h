#ifndef __SeedFinderBase_H__
#define __SeedFinderBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

class SeedFinderBase {
 public:
  SeedFinderBase(const edm::ParameterSet& conf):
    _seedingThreshold(conf.getParameter<double>("seedingThreshold")),
    _algoName(conf.getParameter<std::string>("algoName")) { }
  SeedFinderBase(const SeedFinderBase&) = delete;
  SeedFinderBase& operator=(const SeedFinderBase&) = delete;

  virtual void findSeeds( const edm::Handle<reco::PFRecHitCollection>& input, 
			  const std::vector<bool>& mask,
			  std::vector<bool>& seedable ) = 0;
  
  const std::string& name() const { return _algoName; }

 protected:
  const float _seedingThreshold;  

 private:
  const std::string _algoName;
  
};

#endif
