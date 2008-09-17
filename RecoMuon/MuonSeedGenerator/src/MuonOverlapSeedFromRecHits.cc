#include "RecoMuon/MuonSeedGenerator/src/MuonOverlapSeedFromRecHits.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include <iomanip>


MuonOverlapSeedFromRecHits::MuonOverlapSeedFromRecHits(const edm::EventSetup & eSetup)
: MuonSeedFromRecHits(eSetup)
{
  //FIXME make configurable
  // parameters for the fit of dphi between chambers vs. eta
  // pt = (c1 + c2 abs(eta))/ dphi
  fillConstants(1,4, 1.14, -0.883);
  fillConstants(1,6, 0.782, -0.509);
  fillConstants(1,8, 0.2823, -0.0706);
  fillConstants(2,4, 0.3649, -0.2865);
  fillConstants(2,6, 0.3703, -0.3507);

}


void MuonOverlapSeedFromRecHits::fillConstants(int dtStation, int cscChamberType, double c1, double c2)
{
  theConstantsMap[std::make_pair(dtStation,cscChamberType)] = std::make_pair(c1, c2);
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
      barrelHits.push_back(*iter);
    }
    else
    {
      endcapHits.push_back(*iter);
    }
  }

  for ( MuonRecHitContainer::const_iterator barrelHitItr = barrelHits.begin(),
        lastBarrelHit = barrelHits.end();
        barrelHitItr != lastBarrelHit; ++barrelHitItr)
  {
    for ( MuonRecHitContainer::const_iterator endcapHitItr = endcapHits.begin(),
      lastEndcapHit = endcapHits.end();
      endcapHitItr != lastEndcapHit; ++endcapHitItr)
    {
      TrajectorySeed seed;
      bool good = makeSeed(*barrelHitItr, *endcapHitItr, seed);
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
                                     TrajectorySeed & result) const
{
  DTChamberId dtId(barrelHit->geographicalId().rawId());
  int wheel = dtId.wheel();
  int dtStation = dtId.station();

  //std::cout << "DT " << wheel << " " << dtStation << std::endl; 

  CSCDetId cscId(endcapHit->geographicalId().rawId());
  int cscChamberType = CSCChamberSpecs::whatChamberType(cscId.station(), cscId.ring());
  //std::cout << " CSC " << cscChamberType << std::endl;



  // find the parametrization constants
  std::pair<int, int> key(dtStation, cscChamberType);
  ConstantsMap::const_iterator mapItr = theConstantsMap.find(key);
  if(mapItr != theConstantsMap.end())
  {

    double dphi = (*barrelHit).globalPosition().phi() - (*endcapHit).globalPosition().phi();

    if(dphi > M_PI) dphi -= 2*M_PI;
    if(dphi < -M_PI) dphi += 2*M_PI;
    double eta = (*barrelHit).globalPosition().eta();

    double c1 = mapItr->second.first;
    double c2 = mapItr->second.second;
    // the parametrization
    double pt = (c1 + c2 * fabs(eta) ) / dphi;
    double minpt = 3.;
    float sigmapt = 25.;
    // if too small, probably an error.  Keep trying.
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

    // use the endcap hit, since segments at the edge of the barrel
    // might just be 2D
    result = createSeed(pt, sigmapt, endcapHit);
    //std::cout << "OVERLAPFITTED PT " << pt << " dphi " << dphi << " eta " << eta << std::endl;
    return true;
  }
  return false;
}

