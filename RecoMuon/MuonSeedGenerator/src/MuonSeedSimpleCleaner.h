#ifndef MuonSeedGenerator_MuonSeedSimpleCleaner_h
#define MuonSeedGenerator_MuonSeedSimpleCleaner_h

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedVCleaner.h"

class MuonSeedSimpleCleaner: public MuonSeedVCleaner
{
public:

  virtual void clean(TrajectorySeedCollection & seeds);

private:
  bool checkPt(const TrajectorySeed & seed) const;
};

#endif

