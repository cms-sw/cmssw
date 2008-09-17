#ifndef MuonSeedGenerator_MuonCSCSeedFromRecHits_h
#define MuonSeedGenerator_MuonCSCSeedFromRecHits_h

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFromRecHits.h"
#include <map>

class MuonCSCSeedFromRecHits : public MuonSeedFromRecHits
{
public:

  MuonCSCSeedFromRecHits(const edm::EventSetup & eSetup);
  virtual ~MuonCSCSeedFromRecHits() {}

  virtual TrajectorySeed seed() const;

private:

  void fillConstants(int chamberType1, int chamberType2, double c1, double c2);

  // try to make something from a pair of layers with hits.
  bool makeSeed(const MuonRecHitContainer & hits1, const MuonRecHitContainer & hits2,
                 TrajectorySeed & seed) const;

  // when all else fails
  void makeDefaultSeed(TrajectorySeed & seed) const;

  bool createDefaultEndcapSeed(ConstMuonRecHitPointer last,TrajectorySeed & seed) const;
  float computeDefaultPt(ConstMuonRecHitPointer muon) const;
  int segmentQuality(ConstMuonRecHitPointer muon) const;
  ConstMuonRecHitPointer  bestSegment() const;

  void analyze() const;

  typedef std::map< std::pair<int, int>, std::pair<double, double> > ConstantsMap;
  ConstantsMap theConstantsMap;
};

#endif

