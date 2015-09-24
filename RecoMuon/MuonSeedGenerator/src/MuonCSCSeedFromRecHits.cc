#include "RecoMuon/MuonSeedGenerator/src/MuonCSCSeedFromRecHits.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedPtExtractor.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include <iomanip>


MuonCSCSeedFromRecHits::MuonCSCSeedFromRecHits()
: MuonSeedFromRecHits()
{
}


TrajectorySeed MuonCSCSeedFromRecHits::seed() const
{
  TrajectorySeed result;
  if(theRhits.size() == 1) 
  {
     //return createSeed(100., 100., theRhits[0]);
     makeDefaultSeed(result);
     return result;
  }
  //@@ doesn't handle overlap between ME11 and ME12 correctly
  // sort by station
  MuonRecHitContainer station1Hits, station2Hits, station3Hits, station4Hits;
  for ( MuonRecHitContainer::const_iterator iter = theRhits.begin(), end = theRhits.end();
        iter != end; ++iter)
  {
    int station = CSCDetId((*iter)->geographicalId().rawId()).station();
    if(station == 1)
    {
      station1Hits.push_back(*iter);
    }
    else if(station == 2) 
    {
      station2Hits.push_back(*iter);
    }
    else if(station == 3)
    {
      station3Hits.push_back(*iter);
    }
    else if(station == 4)
    {
      station4Hits.push_back(*iter);
    }
  }

  //std::cout << "Station hits " << station1Hits.size() << " " 
  //                            << station2Hits.size() << " "
  //                             << station3Hits.size() << std::endl;

  // see whether station 2 or station 3 is better
  MuonRecHitContainer * betterSecondHits = &station2Hits; 
  MuonRecHitContainer * notAsGoodSecondHits = &station3Hits;
  if(!station2Hits.empty() && !station3Hits.empty())
  { 
    // swap if station 3 has better quailty
    if(segmentQuality(station3Hits[0]) < segmentQuality(station2Hits[0]))
    {
      betterSecondHits = &station3Hits;
      notAsGoodSecondHits = &station2Hits; 
    }
  }

  // now try to make pairs
  if(makeSeed(station1Hits, *betterSecondHits, result))
  {
    return result;
  }
  if(makeSeed(station1Hits, *notAsGoodSecondHits, result))
  {
    return result;
  }
  if(makeSeed(station2Hits, station3Hits, result))
  {
    return result;
  }
  if(makeSeed(station1Hits, station4Hits, result))
  {
    return result;
  }



  // no luck
  makeDefaultSeed(result);
  return result;
}


bool MuonCSCSeedFromRecHits::makeSeed(const MuonRecHitContainer & hits1, const MuonRecHitContainer & hits2,
                                      TrajectorySeed & seed) const
{
  for ( MuonRecHitContainer::const_iterator itr1 = hits1.begin(), end1 = hits1.end();
        itr1 != end1; ++itr1)
  {
    CSCDetId cscId1((*itr1)->geographicalId().rawId());
    //int type1 = CSCChamberSpecs::whatChamberType(cscId1.station(), cscId1.ring());

    for ( MuonRecHitContainer::const_iterator itr2 = hits2.begin(), end2 = hits2.end();
          itr2 != end2; ++itr2)
    {

      CSCDetId cscId2((*itr2)->geographicalId().rawId());
      //int type2 = CSCChamberSpecs::whatChamberType(cscId2.station(), cscId2.ring());

        // take the first pair that comes along.  Probably want to rank them later
      std::vector<double> pts = thePtExtractor->pT_extract(*itr1, *itr2);
        
        double pt = pts[0];
        double sigmapt = pts[1];
        double minpt = 3.;

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

          // get the position and direction from the higher-quality segment
          ConstMuonRecHitPointer bestSeg = bestEndcapHit(theRhits);
          seed = createSeed(pt, sigmapt, bestSeg);

          //std::cout << "FITTED TIMESPT " << pt << " dphi " << dphi << " eta " << eta << std::endl;
          return true;
        }

    }
  }

  // guess it didn't find one
//std::cout << "NOTHING FOUND ! " << std::endl;
  return false;
}


//typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
int MuonCSCSeedFromRecHits::segmentQuality(ConstMuonRecHitPointer  segment) const
{
  int Nchi2 = 0;
  int quality = 0;
  int nhits = segment->recHits().size();
  if ( segment->chi2()/(nhits*2.-4.) > 3. ) Nchi2 = 1;
  if ( segment->chi2()/(nhits*2.-4.) > 9. ) Nchi2 = 2;

  if ( nhits >  4 ) quality = 1 + Nchi2;
  if ( nhits == 4 ) quality = 3 + Nchi2;
  if ( nhits == 3 ) quality = 5 + Nchi2;

  float dPhiGloDir = fabs ( deltaPhi(segment->globalPosition().phi(), segment->globalDirection().phi()) );

  if ( dPhiGloDir > .2 ) ++quality;
  // add a penalty for being ME1A if the chamber is ganged
  if ( segment->isCSC() and CSCDetId(segment->geographicalId()).ring() == 4 )
  {
    const auto chamber = dynamic_cast<const CSCChamber*>(segment->det());
    if ( chamber->specs()->gangedStrips() ) ++quality;
  }

  return quality;
}



MuonCSCSeedFromRecHits::ConstMuonRecHitPointer
MuonCSCSeedFromRecHits::bestEndcapHit(const MuonRecHitContainer & endcapHits) const
{
  MuonRecHitPointer me1=0, meit=0;
  float dPhiGloDir = .0;                            //  +v
  float bestdPhiGloDir = M_PI;                      //  +v
  int quality1 = 0, quality = 0;        //  +v  I= 5,6-p. / II= 4p.  / III= 3p.

  for ( MuonRecHitContainer::const_iterator iter = endcapHits.begin(); iter!= endcapHits.end(); iter++ ){
    if ( !(*iter)->isCSC() ) continue;

    // tmp compar. Glob-Dir for the same tr-segm:

    meit = *iter;

    quality = segmentQuality(meit);

    dPhiGloDir = fabs ( deltaPhi(meit->globalPosition().phi(), meit->globalDirection().phi()) );

    if(!me1){
      me1 = meit;
      quality1 = quality;
      bestdPhiGloDir = dPhiGloDir;
    }

    if(me1) {

      if ( !me1->isValid() ) {
        me1 = meit;
        quality1 = quality;
        bestdPhiGloDir = dPhiGloDir;
      }

      if ( me1->isValid() && quality < quality1 ) {
        me1 = meit;
        quality1 = quality;
        bestdPhiGloDir = dPhiGloDir;
      }

      if ( me1->isValid() && bestdPhiGloDir > .03 ) {
        if ( dPhiGloDir < bestdPhiGloDir - .01 && quality == quality1 ) {
          me1 = meit;
          quality1 = quality;
          bestdPhiGloDir = dPhiGloDir;
        }
      }
    }

  }   //  iter

  return me1;
}


void MuonCSCSeedFromRecHits::makeDefaultSeed(TrajectorySeed & seed) const
{
  //Search ME1  ...
  ConstMuonRecHitPointer me1= bestEndcapHit(theRhits);
  //  bool good=false;

  if(me1 && me1->isValid() )
  {
    //revert if a LogTrace or smth is necessary    good = 
    createDefaultEndcapSeed(me1, seed); 
  }
}





bool 
MuonCSCSeedFromRecHits::createDefaultEndcapSeed(ConstMuonRecHitPointer last, 
				 TrajectorySeed & seed) const {
  //float momentum = computeDefaultPt(last);
  std::vector<double> momentum = thePtExtractor->pT_extract(last, last);
  seed = createSeed(momentum[0],momentum[1],last);
  return true;
}



void MuonCSCSeedFromRecHits::analyze() const 
{
  for ( MuonRecHitContainer::const_iterator iter = theRhits.begin(); iter!= theRhits.end(); iter++ ){
    if ( !(*iter)->isCSC() ) continue;
    for ( MuonRecHitContainer::const_iterator iter2 = iter + 1; iter2 != theRhits.end(); ++iter2)
    {
      if( !(*iter2)->isCSC() ) continue;

      CSCDetId cscId1((*iter)->geographicalId().rawId());
      CSCDetId cscId2((*iter2)->geographicalId().rawId());
      double dphi = deltaPhi((**iter).globalPosition().phi(), (**iter2).globalPosition().phi());

      int type1 = cscId1.iChamberType();
      int type2 = cscId2.iChamberType();

      // say the lower station first
      if(type1 < type2)
      {
        std::cout << "HITPAIRA," << type1 << type2 << "," <<
            dphi << "," << (**iter).globalPosition().eta() << std::endl;
      }
      if(type2 < type1)
      {
        std::cout << "HITPAIRB," << type2 << type1 << "," <<
            -dphi << "," << (**iter2).globalPosition().eta() << std::endl;
      }
      if(type1 == type2)
      {
        std::cout << "HITPAIRSAMESTATION," << type1 << cscId1.ring() << cscId2.ring()
           << "," << dphi << "," << (**iter).globalPosition().eta() << std::endl;
      }

    }
  }
}
