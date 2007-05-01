#ifndef MuonSeedGenerator_MuonOverlapSeedFromRecHits_h
#define MuonSeedGenerator_MuonOverlapSeedFromRecHits_h

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFromRecHits.h"
#include <map>

class MuonOverlapSeedFromRecHits : public MuonSeedFromRecHits
{
public:

  MuonOverlapSeedFromRecHits(const edm::EventSetup & eSetup);
  virtual ~MuonOverlapSeedFromRecHits() {}

  std::vector<TrajectorySeed> seeds() const;

  bool makeSeed(MuonTransientTrackingRecHit::ConstMuonRecHitPointer barrelHit,
                MuonTransientTrackingRecHit::ConstMuonRecHitPointer endcapHit,
                TrajectorySeed & result) const;


private:

  void fillConstants(int dtStation, int cscChamberType, double c1, double c2);

  // try to make something from a pair of layers with hits.
  bool makeSeed(const MuonRecHitContainer & barrelHits, 
                const MuonRecHitContainer & endcapHits,
                TrajectorySeed & seed) const;

  typedef std::map< std::pair<int, int>, std::pair<double, double> > ConstantsMap;
  ConstantsMap theConstantsMap;
};

#endif

