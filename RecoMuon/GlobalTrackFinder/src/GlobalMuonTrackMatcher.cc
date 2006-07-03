#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackMatcher.h"

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with a tracker track
 *
 *  $Date: 2006/07/02 03:01:37 $
 *  $Revision: 1.1 $
 *  \author Chang Liu  - Purdue University
 */
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


GlobalMuonTrackMatcher::GlobalMuonTrackMatcher(double chi2) : theMaxChi2(chi2) {

}

/** choose the tk Track from a TrackCollection which has smallest Chi2 with
   a given standalone Track
 */
std::pair<bool, reco::TrackRef> 
GlobalMuonTrackMatcher::match(const reco::TrackRef& staT, const edm::Handle<reco::TrackCollection>& tkTs) const {

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
GlobalMuonTrackMatcher::match(const reco::TrackRef&, const reco::TrackRef&) const {

  return(std::pair<bool,double>(false,0));

}

/** determine if two TSOSs are compatible
 */
std::pair<bool,double> 
GlobalMuonTrackMatcher::match(const TrajectoryStateOnSurface& tsos1, const TrajectoryStateOnSurface& tsos2) const {

  AlgebraicVector v(tsos1.globalParameters().vector() - tsos2.globalParameters().vector());
  AlgebraicSymMatrix m(tsos1.curvilinearError().matrix() + tsos2.curvilinearError().matrix());
  int ierr;
  m.invert(ierr);
  // if (ierr != 0) ...
  double est = m.similarity(v);

  return ( est > theMaxChi2 ) ? std::pair<bool,double>(false,est) : std::pair<bool,double>(true,est);

}

