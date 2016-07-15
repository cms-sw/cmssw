#include "PassThruSeedFinder.h"

PassThruSeedFinder::
PassThruSeedFinder(const edm::ParameterSet& conf) : 
  SeedFinderBase(conf){
}

// the starting state of seedable is all false!
void PassThruSeedFinder::
findSeeds( const edm::Handle<reco::PFRecHitCollection>& input,
	   const std::vector<bool>& mask,
	   std::vector<bool>& seedable ) {  
  seedable = std::move(std::vector<bool>(input->size(),true));  
}
