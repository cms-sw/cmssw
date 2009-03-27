#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"

using namespace reco;

ElectronPixelSeed::ElectronPixelSeed( )
 : TrajectorySeed()
{
}

ElectronPixelSeed::ElectronPixelSeed(edm::Ref<SuperClusterCollection>& scl, PTrajectoryStateOnDet & pts, recHitContainer & rh,  PropagationDirection & dir)
: TrajectorySeed(pts,rh,dir)
{

  theSuperCluster = scl;
}

ElectronPixelSeed::ElectronPixelSeed(edm::Ref<SuperClusterCollection>& scl, const TrajectorySeed & seed)
: TrajectorySeed(seed)
{

  theSuperCluster = scl;
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

