/**
 *  A selector for muon tracks
 *
 *  $Date:  $
 *  $Revision: $
 *  \author R.Bellan - INFN Torino
 */
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"

//#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


void MuonTrajectoryCleaner::clean(TrajectoryContainer& trajC){ 

  LogDebug("MuonTrajectoryCleaner") << "****** Muon Trajectory Cleaner ******" << endl;

  TrajectoryContainer::iterator iter, jter;
  Trajectory::DataContainer::const_iterator m1, m2;

  if ( trajC.size() < 2 ) return;

  int i(0), j(0);
  int match(0);

  vector<bool> mask(trajC.size(),true);
  
  TrajectoryContainer result;
  result.reserve(trajC.size());
  
  for ( iter = trajC.begin(); iter != trajC.end(); iter++ ) {
    if ( !mask[i] ) { i++; continue; }
    const Trajectory::DataContainer& meas1 = (*iter).measurements();
    j = i+1;
    bool skipnext=false;
    for ( jter = iter+1; jter != trajC.end(); jter++ ) {
      if ( !mask[j] ) { j++; continue; }
      const Trajectory::DataContainer& meas2 = (*jter).measurements();
      match = 0;
      for ( m1 = meas1.begin(); m1 != meas1.end(); m1++ ) {
        for ( m2 = meas2.begin(); m2 != meas2.end(); m2++ ) {
          if ( (*m1).recHit() == (*m2).recHit() ) match++;
        }
      }
      
      LogDebug("MuonTrajectoryCleaner") 
	<< " MuonTrajSelector: trajC " << i << "chi2/nRH=" 
	<< (*iter).chiSquared() << "/" << (*iter).foundHits() <<
	"vs trajC" << j << "chi2/nRH=" << (*jter).chiSquared() <<
	"/" << (*jter).foundHits() << "Shared RecHits:" << match;
       
      // If there are matches, reject the worst track
      if ( match > 0 ) {
        if (  (*iter).foundHits() == (*jter).foundHits() ) {
          if ( (*iter).chiSquared() > (*jter).chiSquared() ) {
            mask[i] = false;
            skipnext=true;
          }
          else mask[j] = false;
        }
        else { // different number of hits
          if ( (*iter).foundHits() < (*jter).foundHits() ) {
	    mask[i] = false;
            skipnext=true;
          }
          else mask[j] = false;
	}
      }
      if(skipnext) break;
      j++;
    }
    i++;
    if(skipnext) continue;
  }
  
  i = 0;
  for ( iter = trajC.begin(); iter != trajC.end(); iter++ ) {
    if ( mask[i] ) result.push_back(*iter);
    i++;
  }
  
  trajC.clear();
  trajC = result;
}
