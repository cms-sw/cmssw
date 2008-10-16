/**
 *  See header file for a description of this class.
 *
 *  $Date: 2008/09/12 23:09:07 $
 *  $Revision: 1.26 $
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *  \author porting  R. Bellan
 *
 */


#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

  if ( num_bar ) {
    LogDebug(metname)
      << "Barrel Seeds " << num_bar << endl;
    result.push_back(theBarrel.seed());
 
    //if ( debug ) //2
      // cout << result.back().startingState() << endl;
      // was
      // cout << result.back().freeTrajectoryState() << endl;
  }
  
  

  int num_endcap = 0;
  for ( MuonRecHitContainer::const_iterator iter = hits.begin(); iter!= hits.end(); iter++ ){
    if ( (*iter)->isCSC() )
    {
//std::cout << **iter << std::endl;
      theEndcap.add(*iter);
      theOverlap.add(*iter);
      ++num_endcap;
    }
  }
  if(num_endcap > 1 || (num_endcap==1 && num_bar==0))
  {
    LogDebug(metname)
      << "Endcap Seeds " << num_endcap << endl;
    result.push_back(theEndcap.seed());
  }

  if(num_bar > 0 && num_endcap > 0)
  {
    LogTrace(metname) << "Overlap Seed" << endl;
    std::vector<TrajectorySeed> overlapSeeds = theOverlap.seeds();
    result.insert(result.end(), overlapSeeds.begin(), overlapSeeds.end());
  }

}

