#include "RecoMuon/MuonSeedGenerator/src/MuonSeedSimpleCleaner.h"

void MuonSeedSimpleCleaner::clean(TrajectorySeedCollection & seeds)
{
  TrajectorySeedCollection output;
  for(std::vector<TrajectorySeed>::iterator seed = seeds.begin();
      seed != seeds.end(); ++seed){
    int counter =0;
    for(std::vector<TrajectorySeed>::iterator seed2 = seed;
        seed2 != seeds.end(); ++seed2)
      if( seed->startingState().parameters().vector() ==
          seed2->startingState().parameters().vector() )
        ++counter;

    if( counter > 1 ) seeds.erase(seed--);
    else output.push_back(*seed);
  }
}

