#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"

using namespace reco;

ElectronPixelSeed::ElectronPixelSeed( )
 : TrajectorySeed()
{
}

ElectronPixelSeed::ElectronPixelSeed(edm::Ref<SuperClusterCollection>& seed, PTrajectoryStateOnDet & pts, recHitContainer & rh,  PropagationDirection & dir)
: TrajectorySeed(pts,rh,dir)
{

  theSuperCluster = seed;
}

ElectronPixelSeed::ElectronPixelSeed( const ElectronPixelSeed & seed )
 : TrajectorySeed(seed)
 {
  theSuperCluster=seed.theSuperCluster;
}

ElectronPixelSeed & ElectronPixelSeed::operator=( const ElectronPixelSeed & seed )
 {
   TrajectorySeed::operator=(seed) ;
   theSuperCluster = seed.theSuperCluster ;
   return *this ;
 }

ElectronPixelSeed::~ElectronPixelSeed() 
 { }

