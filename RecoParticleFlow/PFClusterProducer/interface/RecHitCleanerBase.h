#ifndef __RecHitCleanerBase_H__
#define __RecHitCleanerBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include <string>

class RecHitCleanerBase {
 public:
  RecHitCleanerBase(const edm::ParameterSet& conf);
  RecHitCleanerBase(const RecHitCleanerBase& ) = delete;
  RecHitCleanerBase& operator=(const RecHitCleanerBase&) = delete;

  virtual void clean(const reco::PFRecHitRefVector&, std::vector<bool>&) = 0;

  const std::string& name() const { return _algoName; }

 private:
  const std::string _algoName;
};

#endif
