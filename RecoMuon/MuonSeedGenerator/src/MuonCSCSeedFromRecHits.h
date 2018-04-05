#ifndef MuonSeedGenerator_MuonCSCSeedFromRecHits_h
#define MuonSeedGenerator_MuonCSCSeedFromRecHits_h

#include "RecoMuon/TrackingTools/interface/MuonSeedFromRecHits.h"

class MuonCSCSeedFromRecHits : public MuonSeedFromRecHits
{
public:

  MuonCSCSeedFromRecHits();
  ~MuonCSCSeedFromRecHits() override {}

  virtual TrajectorySeed seed() const;

  ConstMuonRecHitPointer bestEndcapHit(const MuonRecHitContainer & endcapHits) const;

private:

  // try to make something from a pair of layers with hits.
  bool makeSeed(const MuonRecHitContainer & hits1, const MuonRecHitContainer & hits2,
                 TrajectorySeed & seed) const;
  bool makeSeed2(const MuonRecHitContainer & hits1, const MuonRecHitContainer & hits2,
                 TrajectorySeed & seed) const;

  // when all else fails
  void makeDefaultSeed(TrajectorySeed & seed) const;

  bool createDefaultEndcapSeed(ConstMuonRecHitPointer last,TrajectorySeed & seed) const;
  float computeDefaultPt(ConstMuonRecHitPointer muon) const;
  int segmentQuality(ConstMuonRecHitPointer muon) const;

  void analyze() const;
};

#endif

