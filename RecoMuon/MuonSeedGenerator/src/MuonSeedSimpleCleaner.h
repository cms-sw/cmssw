#ifndef MuonSeedGenerator_MuonSeedSimpleCleaner_h
#define MuonSeedGenerator_MuonSeedSimpleCleaner_h

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedVCleaner.h"

class MuonSeedSimpleCleaner: public MuonSeedVCleaner
{
public:

  void clean(TrajectorySeedCollection & seeds) override;

private:
  bool checkPt(const TrajectorySeed & seed) const;
};

#endif

