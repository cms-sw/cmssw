#include "RecoMuon/MuonSeedGenerator/src/MuonOverlapSeedFromRecHits.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonDTSeedFromRecHits.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonCSCSeedFromRecHits.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedPtExtractor.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include <iomanip>


MuonOverlapSeedFromRecHits::MuonOverlapSeedFromRecHits()
: MuonSeedFromRecHits()
{
}




std::vector<TrajectorySeed> MuonOverlapSeedFromRecHits::seeds() const
{
  std::vector<TrajectorySeed> result;
  //@@ doesn't handle overlap between ME11 and ME12 correctly
  // sort by station
  MuonRecHitContainer barrelHits, endcapHits;
  for ( MuonRecHitContainer::const_iterator iter = theRhits.begin(), end = theRhits.end();
        iter != end; ++iter)
  {
    if((*iter)->isDT())
    {
      DTChamberId dtId((**iter).geographicalId().rawId());
      // try not doing seeds that start in DT station 2, if there'as a good single segment seed
      if(dtId.station() == 1 || (dtId.station()==2 && (*iter)->dimension() == 2))
      {
        barrelHits.push_back(*iter);
      }
    }
    else
    {
      endcapHits.push_back(*iter);
    }
  }

  ConstMuonRecHitPointer bestSegment = bestHit(barrelHits, endcapHits);
  for ( MuonRecHitContainer::const_iterator barrelHitItr = barrelHits.begin(),
        lastBarrelHit = barrelHits.end();
        barrelHitItr != lastBarrelHit; ++barrelHitItr)
  {
    for ( MuonRecHitContainer::const_iterator endcapHitItr = endcapHits.begin(),
      lastEndcapHit = endcapHits.end();
      endcapHitItr != lastEndcapHit; ++endcapHitItr)
    {
      TrajectorySeed seed;
      bool good = makeSeed(*barrelHitItr, *endcapHitItr, bestSegment, seed);
      if(good) result.push_back(seed);
      // try just one seed
      return result;
    }
  }

  //std::cout << "Overlap hits " << barrelHits.size() << " " 
  //                             << endcapHits.size() << std::endl;

  return result;
}



bool
MuonOverlapSeedFromRecHits::makeSeed(MuonTransientTrackingRecHit::ConstMuonRecHitPointer barrelHit,
                                     MuonTransientTrackingRecHit::ConstMuonRecHitPointer endcapHit,
                                     MuonTransientTrackingRecHit::ConstMuonRecHitPointer bestSegment,
                                     TrajectorySeed & result) const
{
  std::vector<double> pts = thePtExtractor->pT_extract(barrelHit, endcapHit);
  double minpt = 3.;
  double pt = pts[0];
  double sigmapt = pts[1];
    // if too small, probably an error.  Keep trying.
  if(pt != 0) {
    if(fabs(pt) > minpt)
    {
      double maxpt = 2000.;
      if(pt > maxpt) {
        pt = maxpt;
        sigmapt = maxpt;
      }
      if(pt < -maxpt) {
        pt = -maxpt;
        sigmapt = maxpt;
      }
    }

    result = createSeed(pt, sigmapt, bestSegment);
    //std::cout << "OVERLAPFITTED PT " << pt << " dphi " << dphi << " eta " << eta << std::endl;
    return true;
  }
  return false;
}

MuonTransientTrackingRecHit::ConstMuonRecHitPointer 
MuonOverlapSeedFromRecHits::bestHit(
  const MuonTransientTrackingRecHit::MuonRecHitContainer & barrelHits, 
  const MuonTransientTrackingRecHit::MuonRecHitContainer & endcapHits) const
{
  MuonDTSeedFromRecHits dtSeeder;
  MuonCSCSeedFromRecHits cscSeeder;

  ConstMuonRecHitPointer result;
  if(barrelHits.size() > endcapHits.size()) 
  {
    result = dtSeeder.bestBarrelHit(barrelHits);
    if (result->dimension() == 2) result = cscSeeder.bestEndcapHit(endcapHits);
  }
  else
  {
    result = cscSeeder.bestEndcapHit(endcapHits);
  }
  return result;
}




