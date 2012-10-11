#include "TMath.h"
#include "DataFormats/MuonReco/interface/MuonCocktails.h"
#include "DataFormats/TrackReco/interface/Track.h"

//
// Return the TeV-optimized refit track (aka the cocktail or Tune P) or
// the tracker track if either the optimized pT or tracker pT is below the pT threshold
//
reco::Muon::MuonTrackTypePair  muon::tevOptimized(const reco::TrackRef& combinedTrack,
						  const reco::TrackRef& trackerTrack,
						  const reco::TrackRef& tpfmsTrack,
						  const reco::TrackRef& pickyTrack,
						  const double ptThreshold,
						  const double tune1,
						  const double tune2) {
  
  // If Tracker pT is below the pT threshold (currently 200 GeV) - return the Tracker track
  if (trackerTrack->pt() < ptThreshold) return make_pair(trackerTrack,reco::Muon::InnerTrack);  
  
  // Array for convenience below.
  const reco::Muon::MuonTrackTypePair refit[4] = { 
    make_pair(trackerTrack, reco::Muon::InnerTrack), 
    make_pair(combinedTrack,reco::Muon::CombinedTrack),
    make_pair(tpfmsTrack,   reco::Muon::TPFMS),
    make_pair(pickyTrack,   reco::Muon::Picky)
  }; 
  
  // Calculate the log(tail probabilities). If there's a problem,
  // signify this with prob == 0. The current problems recognized are:
  // the track being not available, whether the (re)fit failed or it's
  // just not in the event, or if the (re)fit ended up with no valid
  // hits.
  double prob[4] = {0.,0.,0.,0.};
  bool valid[4] = {0,0,0,0};
  for (unsigned int i = 0; i < 4; ++i) 
    if (refit[i].first.isNonnull()){ 
      valid[i] = true;
      if (refit[i].first->numberOfValidHits()) 
	prob[i] = muon::trackProbability(refit[i].first); 
    }

  
  // Start with picky.
  int chosen = 3;
  
  // If there's a problem with picky, make the default one of the
  // other tracks. Try TPFMS first, then global, then tracker-only.
  if (prob[3] == 0.) { 
    if      (prob[2] > 0.) chosen = 2;
    else if (prob[1] > 0.) chosen = 1;
    else if (prob[0] > 0.) chosen = 0;
  } 
  
  // Now the algorithm: switch from picky to tracker-only if the
  // difference, log(tail prob(picky)) - log(tail prob(tracker-only))
  // is greater than a tuned value. Then compare the
  // so-picked track to TPFMS in the same manner using another tuned
  // value.
  if (prob[0] > 0. && prob[3] > 0. && (prob[3] - prob[0]) > tune1)
    chosen = 0;
  if (prob[2] > 0. && (prob[chosen] - prob[2]) > tune2)
    chosen = 2;

  // Sanity checks 
  if (chosen == 3 && !valid[3] ) chosen = 2;
  if (chosen == 2 && !valid[2] ) chosen = 1;
  if (chosen == 1 && !valid[1] ) chosen = 0; 

  // Done. If pT of the chosen track is below the threshold value, return the tracker track.
  if (valid[chosen] && refit[chosen].first->pt() < ptThreshold) return make_pair(trackerTrack,reco::Muon::InnerTrack);    
  
  // Return the chosen track (which can be the global track in
  // very rare cases).
  return refit[chosen];
}

//
// calculate the tail probability (-ln(P)) of a fit
//
double muon::trackProbability(const reco::TrackRef track) {
  
  int nDOF = (int)track->ndof();
  if ( nDOF > 0 && track->chi2()> 0) { 
    return -log(TMath::Prob(track->chi2(), nDOF));
  } else { 
    return 0.0;
  }
  
}

reco::TrackRef muon::getTevRefitTrack(const reco::TrackRef& combinedTrack,
						     const reco::TrackToTrackMap& map) {
  reco::TrackToTrackMap::const_iterator it = map.find(combinedTrack);
  return it == map.end() ? reco::TrackRef() : it->val;
}


//
// Get the sigma-switch decision (tracker-only versus global).
//
reco::Muon::MuonTrackTypePair muon::sigmaSwitch(const reco::TrackRef& combinedTrack,
						const reco::TrackRef& trackerTrack,
						const double nSigma,
						const double ptThreshold) {
  // If either the global or tracker-only fits have pT below threshold
  // (default 200 GeV), return the tracker-only fit.
  if (combinedTrack->pt() < ptThreshold || trackerTrack->pt() < ptThreshold)
    return make_pair(trackerTrack,reco::Muon::InnerTrack);
  
  // If both are above the pT threshold, compare the difference in
  // q/p: if less than two sigma of the tracker-only track, switch to
  // global. Otherwise, use tracker-only.
  const double delta = fabs(trackerTrack->qoverp() - combinedTrack->qoverp());
  const double threshold = nSigma * trackerTrack->qoverpError();
  return delta > threshold ? make_pair(trackerTrack,reco::Muon::InnerTrack) :  make_pair(combinedTrack,reco::Muon::CombinedTrack);
}

//
// Get the TMR decision (tracker-only versus TPFMS).
//
reco::Muon::MuonTrackTypePair muon::TMR(const reco::TrackRef& trackerTrack,
					const reco::TrackRef& fmsTrack,
					const double tune) {
  double probTK  = 0;
  double probFMS = 0;
  
  if (trackerTrack.isNonnull() && trackerTrack->numberOfValidHits())  
    probTK = muon::trackProbability(trackerTrack);
  if (fmsTrack.isNonnull() && fmsTrack->numberOfValidHits())
    probFMS = muon::trackProbability(fmsTrack);
  
  bool TKok  = probTK > 0;
  bool FMSok = probFMS > 0;
  
  if (TKok && FMSok) {
    if (probFMS - probTK > tune)
      return make_pair(trackerTrack,reco::Muon::InnerTrack);
    else
      return  make_pair(fmsTrack,reco::Muon::TPFMS);
  }
  else if (FMSok)
    return  make_pair(fmsTrack,reco::Muon::TPFMS);
  else if (TKok)
    return make_pair(trackerTrack,reco::Muon::InnerTrack);
  else
    return make_pair(reco::TrackRef(),reco::Muon::None); 
}
