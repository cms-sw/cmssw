#include "RecoMuon/MuonSeedGenerator/src/MuonSeedSimpleCleaner.h"

void MuonSeedSimpleCleaner::clean(TrajectorySeedCollection & seeds)
{
  for(std::vector<TrajectorySeed>::iterator seed = seeds.begin();
      seed != seeds.end(); ++seed){
    if(!checkPt(*seed))
    {
       seeds.erase(seed--);
    }
    else
    {
      int counter =0;
      for(std::vector<TrajectorySeed>::iterator seed2 = seed;
          seed2 != seeds.end(); ++seed2)
        if( seed->startingState().parameters().vector() ==
            seed2->startingState().parameters().vector() )
          ++counter;
      if( counter > 1 ) {
         seeds.erase(seed--);
      }
    }
  }
}


bool MuonSeedSimpleCleaner::checkPt(const TrajectorySeed & seed) const
{
  bool result = true;
  if(seed.nHits() == 1)
  {
    int rawId = seed.startingState().detId();
    DetId detId(rawId);

    bool isBarrel = (detId.subdetId() == 1);
    LocalVector p = seed.startingState().parameters().momentum();
    // homemade local-to-global
    double pt = (isBarrel) ? -p.z() : p.perp();
    if(fabs(pt) < 3.05)
    {
      result = false;
    }
  }
  return result;
}

