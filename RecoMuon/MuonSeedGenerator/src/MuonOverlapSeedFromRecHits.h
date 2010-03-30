#ifndef MuonSeedGenerator_MuonOverlapSeedFromRecHits_h
#define MuonSeedGenerator_MuonOverlapSeedFromRecHits_h

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFromRecHits.h"

class MuonOverlapSeedFromRecHits : public MuonSeedFromRecHits
{
public:

  MuonOverlapSeedFromRecHits();
  virtual ~MuonOverlapSeedFromRecHits() {}

  std::vector<TrajectorySeed> seeds() const;

  bool makeSeed(MuonTransientTrackingRecHit::ConstMuonRecHitPointer barrelHit,
                MuonTransientTrackingRecHit::ConstMuonRecHitPointer endcapHit,
                TrajectorySeed & result) const;

};

#endif

