#ifndef __FlagsCleanerECAL_H__
#define __FlagsCleanerECAL_H__

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"

class FlagsCleanerECAL : public RecHitTopologicalCleanerBase {
public:
  FlagsCleanerECAL(const edm::ParameterSet& conf);
  FlagsCleanerECAL(const FlagsCleanerECAL&) = delete;
  FlagsCleanerECAL& operator=(const FlagsCleanerECAL&) = delete;

  // mark rechits which are flagged as one of the values provided in the vector
  void clean(const edm::Handle<reco::PFRecHitCollection>& input, std::vector<bool>& mask) override;

private:
  std::vector<int> v_chstatus_excl_;  // list of rechit status flags to be excluded from seeding
  bool checkFlags(const reco::PFRecHit& hit);
};

DEFINE_EDM_PLUGIN(RecHitTopologicalCleanerFactory, FlagsCleanerECAL, "FlagsCleanerECAL");

#endif
