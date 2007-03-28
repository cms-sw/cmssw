/**
 *  See header file for a description of this class.
 *
 *  $Date: 2006/08/28 16:16:59 $
 *  $Revision: 1.15 $
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *  \author porting  R. Bellan
 *
 */


#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFinder.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonCSCSeedFromRecHits.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonDTSeedFromRecHits.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

MuonSeedFinder::MuonSeedFinder(){
  
  // FIXME put it in a pSet
  // theMinMomentum = pset.getParameter<double>("EndCapSeedMinPt");  //3.0
  theMinMomentum = 3.0;
}


vector<TrajectorySeed> MuonSeedFinder::seeds(const edm::EventSetup& eSetup) const {

  const std::string metname = "Muon|RecoMuon|MuonSeedFinder";

  //  MuonDumper debug;
  vector<TrajectorySeed> theSeeds;

  MuonDTSeedFromRecHits barrel(eSetup);

  int num_bar = 0;
  for ( MuonRecHitContainer::const_iterator iter = theRhits.begin(); iter!= theRhits.end(); iter++ ){
    if ( (*iter)->isDT() ) {
      barrel.add(*iter);
      num_bar++;
    }
  }

  if ( num_bar ) {
    LogDebug(metname)
      << "Barrel Seeds " << num_bar << endl;
    theSeeds.push_back(barrel.seed());
 
    //if ( debug ) //2
      // cout << theSeeds.back().startingState() << endl;
      // was
      // cout << theSeeds.back().freeTrajectoryState() << endl;
  }
  
  
  MuonCSCSeedFromRecHits endcap(eSetup);
  int num_endcap = 0;
  for ( MuonRecHitContainer::const_iterator iter = theRhits.begin(); iter!= theRhits.end(); iter++ ){
    if ( (*iter)->isCSC() )
    {
//std::cout << **iter << std::endl;
      endcap.add(*iter);
      ++num_endcap;
    }
  }

  if(num_endcap)
  {
    LogDebug(metname)
      << "Endcap Seeds " << num_endcap << endl;
    theSeeds.push_back(endcap.seed());
  }

  return theSeeds;
}

