#ifndef MuonSeedGenerator_MuonSeedVCleaner_h
#define MuonSeedGenerator_MuonSeedVCleaner_h

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

class MuonSeedVCleaner
{
public:
  virtual ~MuonSeedVCleaner() {}
  virtual void clean(TrajectorySeedCollection & seeds) = 0;
};

#endif

