/**
 *  See header file for a description of this class.
 *
 *  $Date: 2009/12/03 00:11:45 $
 *  $Revision: 1.28 $
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *  \author porting  R. Bellan
 *
 */


#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
using namespace std;

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

MuonSeedFinder::MuonSeedFinder(const edm::ParameterSet & pset):
  thePtExtractor(pset),
  theBarrel(),
  theOverlap(),
  theEndcap()
{
  
  // FIXME put it in a pSet
  // theMinMomentum = pset.getParameter<double>("EndCapSeedMinPt");  //3.0
  theMinMomentum = 3.0;
  theBarrel.setPtExtractor(&thePtExtractor);
  theOverlap.setPtExtractor(&thePtExtractor);
  theEndcap.setPtExtractor(&thePtExtractor);

}


void MuonSeedFinder::setBField(const MagneticField * field)
{
  theField = field;
  theBarrel.setBField(field);
  theOverlap.setBField(field);
  theEndcap.setBField(field);
}


void MuonSeedFinder::seeds(const MuonTransientTrackingRecHit::MuonRecHitContainer & hits,
                           std::vector<TrajectorySeed> & result)
{
  const std::string metname = "Muon|RecoMuon|MuonSeedFinder";
    
  //  MuonDumper debug;
  theBarrel.clear();
  theOverlap.clear();
  theEndcap.clear();

  int num_bar = 0;
  for ( MuonRecHitContainer::const_iterator iter = hits.begin(); iter!= hits.end(); iter++ ){
    if ( (*iter)->isDT() ) {
      theBarrel.add(*iter);
      theOverlap.add(*iter);
      num_bar++;
    }
  }

  int num_endcap = 0;
  bool hasME1 = false;
  bool found = false;
  for ( MuonRecHitContainer::const_iterator iter = hits.begin(); iter!= hits.end(); iter++ ){
    if ( (*iter)->isCSC() )
    {
//std::cout << **iter << std::endl;
      theEndcap.add(*iter);
      theOverlap.add(*iter);
      if(CSCDetId((*iter)->geographicalId().rawId()).station() == 1) {
        hasME1 = true;
      }
      ++num_endcap;
    }
  }

  if (num_bar > 1) { 
    LogDebug(metname)
      << "Barrel Seeds " << num_bar << endl;
    result.push_back(theBarrel.seed());
    found = true;
  }

  if(num_endcap > 1 && hasME1) {
    LogDebug(metname)
      << "Endcap Seeds " << num_endcap << endl;
    result.push_back(theEndcap.seed());
    found = true;
  }

  // see if we need to do overlap seeds
  if(!found && num_bar > 0 && num_endcap > 0) 
  {
   LogTrace(metname) << "Overlap Seed" << endl;
    std::vector<TrajectorySeed> overlapSeeds = theOverlap.seeds();
    result.insert(result.end(), overlapSeeds.begin(), overlapSeeds.end());
    if(!overlapSeeds.empty()) found = true;
  }

  // Even if there's overlap, maybe add a second seed for 4-D DT segments. 
  // They're that good, and overlaps aren't totally reliable.
  if(num_bar==1 && theBarrel.firstRecHit()->dimension() == 4) {
        LogDebug(metname)
      << "Single Barrel 4D Seeds " << num_bar << endl;
    result.push_back(theBarrel.seed());
    found = true;
  }

  // only do 2D-only seed if desperate
  if(!found && (num_bar==1 && theBarrel.firstRecHit()->dimension() == 2))
  {
    LogDebug(metname)
      << "Single Barrel 2D Seeds " << num_bar << endl;
    result.push_back(theBarrel.seed());
    found = true;
  }

  //  only do single-CSC if really desperate
  if(!found && (num_endcap > 1 || (num_endcap==1 && num_bar==0)))
  {
    LogDebug(metname)
      << "Desperate Endcap Seeds " << num_endcap << endl;
    result.push_back(theEndcap.seed());
    found = true;
  }

}

