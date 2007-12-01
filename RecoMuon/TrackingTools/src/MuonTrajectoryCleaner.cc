/**
 *  A selector for muon tracks
 *
 *  $Date: 2007/05/28 21:46:37 $
 *  $Revision: 1.13 $
 *  \author R.Bellan - INFN Torino
 */
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std; 

void MuonTrajectoryCleaner::clean(TrajectoryContainer& trajC){ 
  const std::string metname = "Muon|RecoMuon|MuonTrajectoryCleaner";

  LogTrace(metname) << "Muon Trajectory Cleaner called" << endl;

  TrajectoryContainer::iterator iter, jter;
  Trajectory::DataContainer::const_iterator m1, m2;

  if ( trajC.size() < 2 ) return;
  
  LogTrace(metname) << "Number of trajectories in the container: " <<trajC.size()<< endl;

  int i(0), j(0);
  int match(0);

  // CAVEAT: vector<bool> is not a vector, its elements are not addressable!
  // This is fine as long as only operator [] is used as in this case.
  // cf. par 16.3.11
  vector<bool> mask(trajC.size(),true);
  
  TrajectoryContainer result;
  //  result.reserve(trajC.size());
  
  for ( iter = trajC.begin(); iter != trajC.end(); iter++ ) {
    if ( !mask[i] ) { i++; continue; }
    const Trajectory::DataContainer& meas1 = (*iter)->measurements();
    j = i+1;
    bool skipnext=false;
    for ( jter = iter+1; jter != trajC.end(); jter++ ) {
      if ( !mask[j] ) { j++; continue; }
      const Trajectory::DataContainer& meas2 = (*jter)->measurements();
      match = 0;
      for ( m1 = meas1.begin(); m1 != meas1.end(); m1++ ) {
        for ( m2 = meas2.begin(); m2 != meas2.end(); m2++ ) {
	  if ( ( (*m1).recHit()->globalPosition() - (*m2).recHit()->globalPosition()).mag()< 10e-5 ) match++;
        }
      }
      
      LogTrace(metname) 
	<< " MuonTrajSelector: trajC " << i << " chi2/nRH = " 
	<< (*iter)->chiSquared() << "/" << (*iter)->foundHits() <<
	" vs trajC " << j << " chi2/nRH = " << (*jter)->chiSquared() <<
	"/" << (*jter)->foundHits() << " Shared RecHits: " << match;
       
      // If there are matches, reject the worst track
      if ( match > 0 ) {
        if (  (*iter)->foundHits() == (*jter)->foundHits() ) {
          if ( (*iter)->chiSquared() > (*jter)->chiSquared() ) {
            mask[i] = false;
            skipnext=true;
          }
          else mask[j] = false;
        }
        else { // different number of hits
          if ( (*iter)->foundHits() < (*jter)->foundHits() ) {
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
    else delete *iter;
    i++;
  }
  
  trajC.clear();
  trajC = result;
}

//
// clean CandidateContainer
//
void MuonTrajectoryCleaner::clean(CandidateContainer& candC){ 
  const std::string metname = "Muon|RecoMuon|MuonTrajectoryCleaner";

  LogTrace(metname) << "Muon Trajectory Cleaner called" << endl;

  if ( candC.size() < 2 ) return;

  CandidateContainer::iterator iter, jter;
  Trajectory::DataContainer::const_iterator m1, m2;

  const float deltaEta = 0.01;
  const float deltaPhi = 0.01;
  const float deltaPt  = 1.0;
  
  LogTrace(metname) << "Number of muon candidates in the container: " <<candC.size()<< endl;

  int i(0), j(0);
  int match(0);
  bool directionMatch = false;

  // CAVEAT: vector<bool> is not a vector, its elements are not addressable!
  // This is fine as long as only operator [] is used as in this case.
  // cf. par 16.3.11
  vector<bool> mask(candC.size(),true);
  
  CandidateContainer result;
  
  for ( iter = candC.begin(); iter != candC.end(); iter++ ) {
    if ( !mask[i] ) { i++; continue; }
    const Trajectory::DataContainer& meas1 = (*iter)->trajectory()->measurements();
    j = i+1;
    bool skipnext=false;

    TrajectoryStateOnSurface innerTSOS;

    if ((*iter)->trajectory()->direction() == alongMomentum) {
      innerTSOS = (*iter)->trajectory()->firstMeasurement().updatedState();
    } 
    else if ((*iter)->trajectory()->direction() == oppositeToMomentum) { 
      innerTSOS = (*iter)->trajectory()->lastMeasurement().updatedState();
    }
    if ( !(innerTSOS.isValid()) ) continue;

    float pt1 = innerTSOS.globalMomentum().perp();
    float eta1 = innerTSOS.globalMomentum().eta();
    float phi1 = innerTSOS.globalMomentum().phi();

    for ( jter = iter+1; jter != candC.end(); jter++ ) {
      if ( !mask[j] ) { j++; continue; }
      directionMatch = false;
      const Trajectory::DataContainer& meas2 = (*jter)->trajectory()->measurements();
      match = 0;
      for ( m1 = meas1.begin(); m1 != meas1.end(); m1++ ) {
        for ( m2 = meas2.begin(); m2 != meas2.end(); m2++ ) {
          if ( (*m1).recHit()->isValid() && (*m2).recHit()->isValid() ) 
	    if ( ( (*m1).recHit()->globalPosition() - (*m2).recHit()->globalPosition()).mag()< 10e-5 ) match++;
        }
      }
      
      LogTrace(metname) 
	<< " MuonTrajSelector: candC " << i << " chi2/nRH = " 
	<< (*iter)->trajectory()->chiSquared() << "/" << (*iter)->trajectory()->foundHits() <<
	" vs trajC " << j << " chi2/nRH = " << (*jter)->trajectory()->chiSquared() <<
	"/" << (*jter)->trajectory()->foundHits() << " Shared RecHits: " << match;

      TrajectoryStateOnSurface innerTSOS2;       
      if ((*jter)->trajectory()->direction() == alongMomentum) {
        innerTSOS2 = (*jter)->trajectory()->firstMeasurement().updatedState();
      }
      else if ((*jter)->trajectory()->direction() == oppositeToMomentum) {
        innerTSOS2 = (*jter)->trajectory()->lastMeasurement().updatedState();
      }
      if ( !(innerTSOS2.isValid()) ) continue;

      float pt2 = innerTSOS2.globalMomentum().perp();
      float eta2 = innerTSOS2.globalMomentum().eta();
      float phi2 = innerTSOS2.globalMomentum().phi();

      float deta(fabs(eta1-eta2));
      float dphi(fabs(Geom::Phi<float>(phi1)-Geom::Phi<float>(phi2)));
      float dpt(abs(pt1-pt2));
      if ( dpt < deltaPt && deta < deltaEta && dphi < deltaPhi ) {
        directionMatch = true;
        LogTrace(metname)
        << " MuonTrajSelector: candC " << i<<" and "<<j<< " direction matched: "
        <<innerTSOS.globalMomentum()<<" and " <<innerTSOS2.globalMomentum();

      }
      
      // If there are matches, reject the worst track
      bool hitsMatch = ((match > 0) && (match/((*iter)->trajectory()->foundHits()) > 0.25) || (match/((*jter)->trajectory()->foundHits()) > 0.25)) ? true : false;
      
      if ( (hitsMatch > 0) || directionMatch ) {
	if (  (*iter)->trajectory()->foundHits() == (*jter)->trajectory()->foundHits() ) {
          if ( (*iter)->trajectory()->chiSquared() > (*jter)->trajectory()->chiSquared() ) {
            mask[i] = false;
            skipnext=true;
          }
          else mask[j] = false;
        }
        else { // different number of hits
          if ( (*iter)->trajectory()->foundHits() < (*jter)->trajectory()->foundHits() ) {
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
  for ( iter = candC.begin(); iter != candC.end(); iter++ ) {
    if ( mask[i] ) result.push_back(*iter);
    i++;
  }
  
  candC.clear();
  candC = result;
}
