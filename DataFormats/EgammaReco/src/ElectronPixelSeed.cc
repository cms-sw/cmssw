#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"

using namespace reco;

ElectronPixelSeed::ElectronPixelSeed()
 : TrajectorySeed(), subDet2_(0), dRz2_(std::numeric_limits<float>::infinity()), dPhi2_(std::numeric_limits<float>::infinity())
{
}

ElectronPixelSeed::ElectronPixelSeed(edm::Ref<SuperClusterCollection>& scl, PTrajectoryStateOnDet & pts, recHitContainer & rh, PropagationDirection & dir,
  int subDet2, float dRz2, float dPhi2 )
: TrajectorySeed(pts,rh,dir), theSuperCluster(scl), subDet2_(subDet2), dRz2_(dRz2), dPhi2_(dPhi2)
{
  //theSuperCluster = scl; ?????
}

ElectronPixelSeed::ElectronPixelSeed( edm::Ref<SuperClusterCollection>& scl, const TrajectorySeed & seed,
  int subDet2, float dRz2, float dPhi2 )
: TrajectorySeed(seed), theSuperCluster(scl), subDet2_(subDet2), dRz2_(dRz2), dPhi2_(dPhi2)
{
  //theSuperCluster = scl; ?????
}

//ElectronPixelSeed::ElectronPixelSeed( const ElectronPixelSeed & seed )
// : TrajectorySeed(seed)
// {?????
//  theSuperCluster=seed.theSuperCluster;
//}
//
//ElectronPixelSeed & ElectronPixelSeed::operator=( const ElectronPixelSeed & seed )
// {?????
//   TrajectorySeed::operator=(seed) ;
//   theSuperCluster = seed.theSuperCluster ;
//   return *this ;
 //}

ElectronPixelSeed::~ElectronPixelSeed() 
 { }

