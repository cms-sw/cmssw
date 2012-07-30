#include "TMath.h"
#include "DataFormats/MuonReco/interface/MuonCocktails.h"
#include "DataFormats/TrackReco/interface/Track.h"

//
// Return the TeV-optimized refit track, aka the cocktail or Tune P.
//
reco::TrackRef muon::tevOptimized(const reco::TrackRef& combinedTrack,
				  const reco::TrackRef& trackerTrack,
				  const reco::TrackRef& tpfmsTrack,
				  const reco::TrackRef& pickyTrack,
				  const double tune1,
				  const double tune2) {
  // Array for convenience below.
  const reco::TrackRef refit[4] = { 
    trackerTrack, 
    combinedTrack, 
    tpfmsTrack, 
    pickyTrack 
  }; 

  // Calculate the log(tail probabilities). If there's a problem,
  // signify this with prob == 0. The current problems recognized are:
  // the track being not available, whether the (re)fit failed or it's
  // just not in the event, or if the (re)fit ended up with no valid
  // hits.
  double prob[4] = {0.};
  for (unsigned int i = 0; i < 4; ++i) 
    if (refit[i].isNonnull() && refit[i]->numberOfValidHits()) 
      prob[i] = muon::trackProbability(refit[i]); 

  //std::cout << "Probabilities: " << prob[0] << " " << prob[1] << " " << prob[2] << " " << prob[3] << std::endl;

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
  // is greater than a tuned value (currently 30). Then compare the
  // so-picked track to TPFMS in the same manner using another tuned
  // value.
  if (prob[0] > 0. && prob[3] > 0. && (prob[3] - prob[0]) > tune1)
    chosen = 0;
  if (prob[2] > 0. && (prob[chosen] - prob[2]) > tune2)
    chosen = 2;

  // Done. Return the chosen track (which can be the global track in
  // very rare cases).
  return refit[chosen];
}

//
// Return the TeV-optimized refit track (older, deprecated version).
//
reco::TrackRef muon::tevOptimizedOld( const reco::TrackRef& combinedTrack,
				      const reco::TrackRef& trackerTrack,
				      const reco::TrackToTrackMap tevMap1,
				      const reco::TrackToTrackMap tevMap2,
				      const reco::TrackToTrackMap tevMap3 ) {

  std::vector<reco::TrackRef> refit(4);
  reco::TrackRef result;
  bool ok[4];
  ok[0] = true; // Assume tracker track OK.
  
  reco::TrackToTrackMap::const_iterator gmrTrack = tevMap1.find(combinedTrack);
  reco::TrackToTrackMap::const_iterator fmsTrack = tevMap2.find(combinedTrack);
  reco::TrackToTrackMap::const_iterator pmrTrack = tevMap3.find(combinedTrack);

  ok[1] = gmrTrack != tevMap1.end();
  ok[2] = fmsTrack != tevMap2.end();
  ok[3] = pmrTrack != tevMap3.end();

  double prob[4];

  if (ok[0]) refit[0] = trackerTrack;
  if (ok[1]) refit[1] = (*gmrTrack).val;
  if (ok[2]) refit[2] = (*fmsTrack).val;
  if (ok[3]) refit[3] = (*pmrTrack).val;
  
  for (unsigned int i=0; i<4; i++)
    prob[i] = (ok[i] && refit[i]->numberOfValidHits())
      ? trackProbability(refit[i]) : 0.0; 

//  std::cout << "Probabilities: " << prob[0] << " " << prob[1] << " " << prob[2] << " " << prob[3] << std::endl;

  if (prob[1] ) result = refit[1];
  if ((prob[1] == 0) && prob[3]) result = refit[3];
  
  if (prob[1] && prob[3] && ((prob[1] - prob[3]) > 0.05 ))  result = refit[3];

  if (prob[0] && prob[2] && fabs(prob[2] - prob[0]) > 30.) {
    result = refit[0];
    return result;
  }

  if ((prob[1] == 0) && (prob[3] == 0) && prob[2]) result = refit[2];

  reco::TrackRef tmin;
  double probmin = 0.0;

  if (prob[1] && prob[3]) {
    probmin = prob[3]; tmin = refit[3];
    if ( prob[1] < prob[3] ) { probmin = prob[1]; tmin = refit[1]; }
  } else if ((prob[3] == 0) && prob[1]) { 
    probmin = prob[1]; tmin = refit[1]; 
  } else if ((prob[1] == 0) && prob[3]) {
    probmin = prob[3]; tmin = refit[3]; 
  }

  if (probmin && prob[2] && ( (probmin - prob[2]) > 3.5 )) {
    result = refit[2];
  }

  return result;
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

//
// Get the sigma-switch decision (tracker-only versus global).
//
reco::TrackRef muon::sigmaSwitch(const reco::TrackRef& combinedTrack,
				 const reco::TrackRef& trackerTrack,
				 const double nSigma,
				 const double ptThreshold) {
  // If either the global or tracker-only fits have pT below threshold
  // (default 200 GeV), return the tracker-only fit.
  if (combinedTrack->pt() < ptThreshold || trackerTrack->pt() < ptThreshold)
    return trackerTrack;
  
  // If both are above the pT threshold, compare the difference in
  // q/p: if less than two sigma of the tracker-only track, switch to
  // global. Otherwise, use tracker-only.
  const double delta = fabs(trackerTrack->qoverp() - combinedTrack->qoverp());
  const double threshold = nSigma * trackerTrack->qoverpError();
  return delta > threshold ? trackerTrack : combinedTrack;
}

//
// Get the TMR decision (tracker-only versus TPFMS).
//
reco::TrackRef muon::TMR(const reco::TrackRef& trackerTrack,
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
      return trackerTrack;
    else
      return fmsTrack;
  }
  else if (FMSok)
    return fmsTrack;
  else if (TKok)
    return trackerTrack;
  else
    return reco::TrackRef();
}
