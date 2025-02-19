#ifndef MuonSeedGenerator_MuonOverlapSeedFromRecHits_h
#define MuonSeedGenerator_MuonOverlapSeedFromRecHits_h

#include "RecoMuon/TrackingTools/interface/MuonSeedFromRecHits.h"

class MuonOverlapSeedFromRecHits : public MuonSeedFromRecHits
{
public:

  MuonOverlapSeedFromRecHits();
  virtual ~MuonOverlapSeedFromRecHits() {}

  std::vector<TrajectorySeed> seeds() const;

  bool makeSeed(MuonTransientTrackingRecHit::ConstMuonRecHitPointer barrelHit,
                MuonTransientTrackingRecHit::ConstMuonRecHitPointer endcapHit,
                MuonTransientTrackingRecHit::ConstMuonRecHitPointer bestSegment,
                TrajectorySeed & result) const;

private:
  ConstMuonRecHitPointer bestHit(
    const MuonRecHitContainer & barrelHits,
    const MuonRecHitContainer & endcapHits) const;

};

#endif

