#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackMatcher.h"

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with a tracker track
 *
 *  $Date: $
 *  $Revision:  $
 *  \author Chang Liu  - Purdue University
 */
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


GlobalMuonTrackMatcher::GlobalMuonTrackMatcher(const double& chi2) {

  theMaxChi2 = chi2;

}

/** choose the tk Track from a TrackCollection which has smallest Chi2 with
   a given standalone Track
 */
std::pair<bool, reco::TrackRef> 
GlobalMuonTrackMatcher::match(const reco::TrackRef& staT, const edm::Handle<reco::TrackCollection>& tkTs){
  bool hasMatchTk = false;
  reco::TrackRef result;
  double minChi2 = theMaxChi2;
  for (int position = 0; position < int(tkTs->size()); position++) {
    reco::TrackRef tkTRef(tkTs,position);
    std::pair<bool,double> check = match(staT,tkTRef);
    if (!check.first) continue;
    hasMatchTk = true;
    if (check.second < minChi2) { 
      minChi2 = check.second;
      result = tkTRef;
    }
  } 
  return(std::pair<bool, reco::TrackRef>(hasMatchTk, result));
}

/** determine if the tk and standalone Tracks are compatible
  by comparing their TSOSs on outer Tracker surface
 */
std::pair<bool,double> 
GlobalMuonTrackMatcher::match(const reco::TrackRef&, const reco::TrackRef&){

  return(std::pair<bool,double>(false,0));

}

/** determine if two TSOSs are compatible
 */
std::pair<bool,double> 
GlobalMuonTrackMatcher::match(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&){

  return(std::pair<bool,double>(false,0));

}

