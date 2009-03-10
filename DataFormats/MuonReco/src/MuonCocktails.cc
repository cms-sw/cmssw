#include "DataFormats/MuonReco/interface/MuonCocktails.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <TROOT.h>

//
// return the TeV-optimized refit track
//
reco::TrackRef muon::tevOptimized( const reco::TrackRef& combinedTrack,
				   const reco::TrackRef& trackerTrack,
                                   const reco::TrackToTrackMap tevMap1,
                                   const reco::TrackToTrackMap tevMap2,
                                   const reco::TrackToTrackMap tevMap3 ) {

  std::vector<reco::TrackRef> refit(4);
  bool ok[4];
  ok[0] = true; // Assume tracker track OK.

  reco::TrackToTrackMap::const_iterator gmrTrack = tevMap1.find(combinedTrack);
  reco::TrackToTrackMap::const_iterator fmsTrack = tevMap2.find(combinedTrack);
  reco::TrackToTrackMap::const_iterator pmrTrack = tevMap3.find(combinedTrack);

  ok[1] = gmrTrack != tevMap1.end();
  ok[2] = fmsTrack != tevMap2.end();
  ok[3] = pmrTrack != tevMap3.end();

  double prob[4];
  int chosen=3;

  if (ok[0]) refit[0] = trackerTrack;
  if (ok[1]) refit[1] = (*gmrTrack).val;
  if (ok[2]) refit[2] = (*fmsTrack).val;
  if (ok[3]) refit[3] = (*pmrTrack).val;
  
  for (unsigned int i=0; i<4; i++)
    prob[i] = (ok[i] && refit[i]->numberOfValidHits())
      ? trackProbability(refit[i]) : 0.0; 

//  std::cout << "Probabilities: " << prob[0] << " " << prob[1] << " " << prob[2] << " " << prob[3] << std::endl;

  if (!prob[3])
    if (prob[2]) chosen=2; else
      if (prob[1]) chosen=1; else
        if (prob[0]) chosen=0;

  if ( prob[0] && prob[3] && ((prob[3]-prob[0]) > 48.) ) chosen=0;
  if ( prob[0] && prob[1] && ((prob[1]-prob[0]) < 3.) ) chosen=1;
  if ( prob[2] && ((prob[chosen]-prob[2]) > 9.) ) chosen=2;
    
  return refit.at(chosen);
}

//
// return the TeV-optimized refit track (older version)
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
// return the TeV-optimized refit track for the muon
//
reco::TrackRef muon::tevOptimized( const reco::Muon& muon,
                                   const reco::TrackToTrackMap tevMap1,
                                   const reco::TrackToTrackMap tevMap2,
                                   const reco::TrackToTrackMap tevMap3 ) {
  return muon::tevOptimized(muon.combinedMuon(), muon.track(),
			    tevMap1, tevMap2, tevMap3);
}

//
// return the TeV-optimized refit track for the muon (older version)
//
reco::TrackRef muon::tevOptimizedOld( const reco::Muon& muon,
				      const reco::TrackToTrackMap tevMap1,
				      const reco::TrackToTrackMap tevMap2,
				      const reco::TrackToTrackMap tevMap3 ) {
  return muon::tevOptimizedOld(muon.combinedMuon(), muon.track(),
			       tevMap1, tevMap2, tevMap3);
}

//
// calculate the tail probability (-ln(P)) of a fit
//
double muon::trackProbability(const reco::TrackRef track) {

  int nDOF = (int)track->ndof();
  if ( nDOF > 0 && track->chi2()> 0) { 
    return -LnChiSquaredProbability(track->chi2(), nDOF);
  } else { 
    return 0.0;
  }

}


