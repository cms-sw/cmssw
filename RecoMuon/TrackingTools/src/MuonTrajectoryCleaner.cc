/**
 *  A selector for muon tracks
 *
 *  $Date: 2006/08/16 10:07:11 $
 *  $Revision: 1.6 $
 *  \author R.Bellan - INFN Torino
 */
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std; 

void MuonTrajectoryCleaner::clean(TrajectoryContainer& trajC){ 
  const std::string metname = "Muon|RecoMuon|MuonTrajectoryCleaner";

  LogDebug(metname) << "Muon Trajectory Cleaner called" << endl;

  TrajectoryContainer::iterator iter, jter;
  Trajectory::DataContainer::const_iterator m1, m2;

  if ( trajC.size() < 2 ) return;
  
  LogDebug(metname) << "Number of trajectories in the container: " <<trajC.size()<< endl;

  int i(0), j(0);
  int match(0);

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
          if ( (*m1).recHit() == (*m2).recHit() ) match++;
        }
      }
      
      LogDebug(metname) 
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
    i++;
  }
  
  trajC.clear();
  trajC = result;
}

//
// check for ghosts
//
void MuonTrajectoryCleaner::checkGhosts(CandidateContainer& candidates){

  if ( candidates.size() < 2 ) return;

  const float deltaEta = 0.01;
  const float deltaPhi = 0.01;
  const float deltaPt  = 1.0;

  CandidateContainer::iterator e = candidates.end();
  CandidateContainer::iterator i1;
  CandidateContainer::iterator i2; 

  for ( i1 = candidates.begin(); i1 != e; ++i1 ) {
    if ( *i1 == 0 ) continue;
    TrajectoryStateOnSurface innerTSOS;
  
    if ((*i1)->trajectory()->direction() == alongMomentum) {
      innerTSOS = (*i1)->trajectory()->firstMeasurement().updatedState();
    } 
    else if ((*i1)->trajectory()->direction() == oppositeToMomentum) { 
      innerTSOS = (*i1)->trajectory()->lastMeasurement().updatedState();
    }
    if ( !(innerTSOS.isValid()) ) continue;

    float pt1 = innerTSOS.globalMomentum().perp();
    float eta1 = innerTSOS.globalMomentum().eta();
    float phi1 = innerTSOS.globalMomentum().phi();
    for ( i2 = i1+1; i2 != e; ++i2 ) {
      if ( *i2 == 0 || *i1 == 0 ) continue;
      TrajectoryStateOnSurface innerTSOS2;

      if ((*i2)->trajectory()->direction() == alongMomentum) {
        innerTSOS2 = (*i2)->trajectory()->firstMeasurement().updatedState();
      }
      else if ((*i2)->trajectory()->direction() == oppositeToMomentum) {
        innerTSOS2 = (*i2)->trajectory()->lastMeasurement().updatedState();
      }
      if ( !(innerTSOS2.isValid()) ) continue;

      float pt2 = innerTSOS2.globalMomentum().perp();
      float eta2 = innerTSOS2.globalMomentum().eta();
      float phi2 = innerTSOS2.globalMomentum().phi();

      float deta(fabs(eta1-eta2));
      float dphi(fabs(Geom::Phi<float>(phi1)-Geom::Phi<float>(phi2)));
      float dpt(abs(pt1-pt2));
      if ( dpt < deltaPt && deta < deltaEta && dphi < deltaPhi ) {
        CandidateContainer::iterator bad;
        if ((*i1)->trajectory()->foundHits() == (*i2)->trajectory()->foundHits() ) 
           bad = ( (*i1)->trajectory()->chiSquared() < (*i2)->trajectory()->chiSquared() ) ? i2 : i1;
        else bad = ( (*i1)->trajectory()->foundHits() > (*i2)->trajectory()->foundHits() ) ? i2 : i1;
        delete (*bad);
        *bad = 0;
      }
    }
  }

  candidates.erase(remove(candidates.begin(),
                          candidates.end(),
                          static_cast<MuonCandidate*>(0)),
                   candidates.end());
}
