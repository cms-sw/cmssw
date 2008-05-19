#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/Track.h"

unsigned int muon::RequiredStationMask( const reco::Muon& muon,
					  double maxChamberDist,
					  double maxChamberDistPull,
					  reco::Muon::ArbitrationType arbitrationType )
{
   unsigned int theMask = 0;

   for(int stationIdx = 1; stationIdx < 5; ++stationIdx)
      for(int detectorIdx = 1; detectorIdx < 3; ++detectorIdx)
         if(muon.trackDist(stationIdx,detectorIdx,arbitrationType) < maxChamberDist &&
               muon.trackDist(stationIdx,detectorIdx,arbitrationType)/muon.trackDistErr(stationIdx,detectorIdx,arbitrationType) < maxChamberDistPull)
            theMask += 1<<((stationIdx-1)+4*(detectorIdx-1));

   return theMask;
}

// ------------ method to calculate the calo compatibility for a track with matched muon info  ------------
float muon::caloCompatibility(const reco::Muon& muon) {
  return muon.caloCompatibility();
}

// ------------ method to calculate the segment compatibility for a track with matched muon info  ------------
float muon::segmentCompatibility(const reco::Muon& muon) {
  bool use_weight_regain_at_chamber_boundary = true;
  bool use_match_dist_penalty = true;

  int nr_of_stations_crossed = 0;
  int nr_of_stations_with_segment = 0;
  std::vector<int> stations_w_track(8);
  std::vector<int> station_has_segmentmatch(8);
  std::vector<int> station_was_crossed(8);
  std::vector<float> stations_w_track_at_boundary(8);
  std::vector<float> station_weight(8);
  int position_in_stations = 0;
  float full_weight = 0.;

  for(int i = 1; i<=8; ++i) {
    // ********************************************************;
    // *** fill local info for this muon (do some counting) ***;
    // ************** begin ***********************************;
    if(i<=4) { // this is the section for the DTs
      if( muon.trackDist(i,1) < 999999 ) { //current "raw" info that a track is close to a chamber
	++nr_of_stations_crossed;
	station_was_crossed[i-1] = 1;
	if(muon.trackDist(i,1) > -10. ) stations_w_track_at_boundary[i-1] = muon.trackDist(i,1); 
	else stations_w_track_at_boundary[i-1] = 0.;
      }
      if( muon.segmentX(i,1) < 999999 ) { //current "raw" info that a segment is matched to the current track
	++nr_of_stations_with_segment;
	station_has_segmentmatch[i-1] = 1;
      }
    }
    else     { // this is the section for the CSCs
      if( muon.trackDist(i-4,2) < 999999 ) { //current "raw" info that a track is close to a chamber
	++nr_of_stations_crossed;
	station_was_crossed[i-1] = 1;
	if(muon.trackDist(i-4,2) > -10. ) stations_w_track_at_boundary[i-1] = muon.trackDist(i-4,2);
	else stations_w_track_at_boundary[i-1] = 0.;
      }
      if( muon.segmentX(i-4,2) < 999999 ) { //current "raw" info that a segment is matched to the current track
	++nr_of_stations_with_segment;
	station_has_segmentmatch[i-1] = 1;
      }
    }
    // rough estimation of chamber border efficiency (should be parametrized better, this is just a quick guess):
    // TF1 * merf = new TF1("merf","-0.5*(TMath::Erf(x/6.)-1)",-100,100);
    // use above value to "unpunish" missing segment if close to border, i.e. rather than not adding any weight, add
    // the one from the function. Only for dist ~> -10 cm, else full punish!.

    // ********************************************************;
    // *** fill local info for this muon (do some counting) ***;
    // ************** end *************************************;
  }

  // ********************************************************;
  // *** calculate weights for each station *****************;
  // ************** begin ***********************************;
  //    const float slope = 0.5;
  //    const float attenuate_weight_regain = 1.;
  // if attenuate_weight_regain < 1., additional punishment if track is close to boundary and no segment
  const float attenuate_weight_regain = 0.5; 

  for(int i = 1; i<=8; ++i) { // loop over all possible stations

    // first set all weights if a station has been crossed
    // later penalize if a station did not have a matching segment

    //old logic      if(station_has_segmentmatch[i-1] > 0 ) { // the track has an associated segment at the current station
    if( station_was_crossed[i-1] > 0 ) { // the track crossed this chamber (or was nearby)
      // - Apply a weight depending on the "depth" of the muon passage. 
      // - The station_weight is later reduced for stations with badly matched segments. 
      // - Even if there is no segment but the track passes close to a chamber boundary, the
      //   weight is set non zero and can go up to 0.5 of the full weight if the track is quite
      //   far from any station.
      ++position_in_stations;

      switch ( nr_of_stations_crossed ) { // define different weights depending on how many stations were crossed
      case 1 : 
	station_weight[i-1] =  1.;
	break;
      case 2 :
	if     ( position_in_stations == 1 ) station_weight[i-1] =  0.33;
	else                                 station_weight[i-1] =  0.67;
	break;
      case 3 : 
	if     ( position_in_stations == 1 ) station_weight[i-1] =  0.23;
	else if( position_in_stations == 2 ) station_weight[i-1] =  0.33;
	else                                 station_weight[i-1] =  0.44;
	break;
      case 4 : 
	if     ( position_in_stations == 1 ) station_weight[i-1] =  0.10;
	else if( position_in_stations == 2 ) station_weight[i-1] =  0.20;
	else if( position_in_stations == 3 ) station_weight[i-1] =  0.30;
	else                                 station_weight[i-1] =  0.40;
	break;
	  
      default : 
// 	LogTrace("MuonIdentification")<<"            // Message: A muon candidate track has more than 4 stations with matching segments.";
// 	LogTrace("MuonIdentification")<<"            // Did not expect this - please let me know: ibloch@fnal.gov";
	// for all other cases
	station_weight[i-1] = 1./nr_of_stations_crossed;
      }

      if( use_weight_regain_at_chamber_boundary ) { // reconstitute some weight if there is no match but the segment is close to a boundary:
	if(station_has_segmentmatch[i-1] <= 0 && stations_w_track_at_boundary[i-1] != 0. ) {
	  // if segment is not present but track in inefficient region, do not count as "missing match" but add some reduced weight. 
	  // original "match weight" is currently reduced by at least attenuate_weight_regain, variing with an error function down to 0 if the track is 
	  // inside the chamber.
	  station_weight[i-1] = station_weight[i-1]*attenuate_weight_regain*0.5*(TMath::Erf(stations_w_track_at_boundary[i-1]/6.)+1.); // remark: the additional scale of 0.5 normalizes Err to run from 0 to 1 in y
	}
	else if(station_has_segmentmatch[i-1] <= 0 && stations_w_track_at_boundary[i-1] == 0.) { // no segment match and track well inside chamber
	  // full penalization
	  station_weight[i-1] = 0.;
	}
      }
      else { // always fully penalize tracks with no matching segment, whether the segment is close to the boundary or not.
	if(station_has_segmentmatch[i-1] <= 0) station_weight[i-1] = 0.;
      }

      if( station_has_segmentmatch[i-1] > 0 && 42 == 42 ) { // if track has matching segment, but the matching is not high quality, penalize
	if(i<=4) { // we are in the DTs
	  if( muon.dY(i,1) != 999999 ) { // have both X and Y match
	    if(
	       TMath::Sqrt(TMath::Power(muon.pullX(i,1),2.)+TMath::Power(muon.pullY(i,1),2.))> 1. ) {
	      // reduce weight
	      if(use_match_dist_penalty) {
		// only use pull if 3 sigma is not smaller than 3 cm
		if(TMath::Sqrt(TMath::Power(muon.dX(i,1),2.)+TMath::Power(muon.dY(i,1),2.)) < 3. && TMath::Sqrt(TMath::Power(muon.pullX(i,1),2.)+TMath::Power(muon.pullY(i,1),2.)) > 3. ) { 
		  station_weight[i-1] *= 1./TMath::Power(
							 TMath::Max((double)TMath::Sqrt(TMath::Power(muon.dX(i,1),2.)+TMath::Power(muon.dY(i,1),2.)),(double)1.),.25); 
		}
		else {
		  station_weight[i-1] *= 1./TMath::Power(
							 TMath::Sqrt(TMath::Power(muon.pullX(i,1),2.)+TMath::Power(muon.pullY(i,1),2.)),.25); 
		}
	      }
	    }
	  }
	  else { // has no match in Y
	    if( muon.pullX(i,1) > 1. ) { // has a match in X
	      // reduce weight
	      if(use_match_dist_penalty) {
		// only use pull if 3 sigma is not smaller than 3 cm
		if( muon.dX(i,1) < 3. && muon.pullX(i,1) > 3. ) { 
		  station_weight[i-1] *= 1./TMath::Power(TMath::Max((double)muon.dX(i,1),(double)1.),.25);
		}
		else {
		  station_weight[i-1] *= 1./TMath::Power(muon.pullX(i,1),.25);
		}
	      }
	    }
	  }
	}
	else { // We are in the CSCs
	  if(
	     TMath::Sqrt(TMath::Power(muon.pullX(i-4,2),2.)+TMath::Power(muon.pullY(i-4,2),2.)) > 1. ) {
	    // reduce weight
	    if(use_match_dist_penalty) {
	      // only use pull if 3 sigma is not smaller than 3 cm
	      if(TMath::Sqrt(TMath::Power(muon.dX(i-4,2),2.)+TMath::Power(muon.dY(i-4,2),2.)) < 3. && TMath::Sqrt(TMath::Power(muon.pullX(i-4,2),2.)+TMath::Power(muon.pullY(i-4,2),2.)) > 3. ) { 
		station_weight[i-1] *= 1./TMath::Power(
						       TMath::Max((double)TMath::Sqrt(TMath::Power(muon.dX(i-4,2),2.)+TMath::Power(muon.dY(i-4,2),2.)),(double)1.),.25);
	      }
	      else {
		station_weight[i-1] *= 1./TMath::Power(
						       TMath::Sqrt(TMath::Power(muon.pullX(i-4,2),2.)+TMath::Power(muon.pullY(i-4,2),2.)),.25);
	      }
	    }
	  }
	}
      }
	
      // Thoughts:
      // - should penalize if the segment has only x OR y info
      // - should also use the segment direction, as it now works!
	
    }
    else { // track did not pass a chamber in this station - just reset weight
      station_weight[i-1] = 0.;
    }
      
    //increment final weight for muon:
    full_weight += station_weight[i-1];
  }

  // if we don't expect any matches, we set the compatibility to
  // 0.5 as the track is as compatible with a muon as it is with
  // background - we should maybe rather set it to -0.5!
  if( nr_of_stations_crossed == 0 ) {
    //      full_weight = attenuate_weight_regain*0.5;
    full_weight = 0.5;
  }

  // ********************************************************;
  // *** calculate weights for each station *****************;
  // ************** end *************************************;

  return full_weight;

}

bool muon::isGoodMuon( const reco::Muon& muon, 
			 AlgorithmType type,
			 double minCompatibility ) {
  if (!muon.isMatchesValid()) return false;
  bool goodMuon = false;
  
  switch( type ) {
  case TM2DCompatibility:
    // Simplistic first cut in the 2D segment- vs calo-compatibility plane. Will have to be refined!
    if( ( caloCompatibility( muon )+segmentCompatibility( muon ) ) > minCompatibility ) goodMuon = true;
    else goodMuon = false;
    return goodMuon;
    break;
  default : 
    // 	LogTrace("MuonIdentification")<<"            // Invalid Algorithm Type called!";
    goodMuon = false;
    return goodMuon;
    break;
  }
}

bool muon::isGoodMuon( const reco::Muon& muon,
			 AlgorithmType type,
			 int minNumberOfMatches,
			 double maxAbsDx,
			 double maxAbsPullX,
			 double maxAbsDy,
			 double maxAbsPullY,
			 double maxChamberDist,
			 double maxChamberDistPull,
			 reco::Muon::ArbitrationType arbitrationType )
{
   if (!muon.isMatchesValid()) return false;
   bool goodMuon = false;

   unsigned int theStationMask = muon.stationMask(arbitrationType);
   unsigned int theRequiredStationMask = RequiredStationMask(muon, maxChamberDist, maxChamberDistPull, arbitrationType);

   // If there are no required stations, require there be at least two segments
   int numSegs = 0;
   for(int it = 0; it < 8; ++it)
      if(theStationMask & 1<<it) ++numSegs;

   if(numSegs > 1) goodMuon = 1;

   // Require that last required station have segment
   if(theRequiredStationMask)
      for(int stationIdx = 7; stationIdx >= 0; --stationIdx)
         if(theRequiredStationMask & 1<<stationIdx)
            if(theStationMask & 1<<stationIdx) {
               goodMuon &= 1;
               break;
            } else {
               goodMuon = false;
               break;
            }

   if(!goodMuon) return false;

   // Impose pull cuts on last segment
   int lastSegBit = 0;
   for(int stationIdx = 7; stationIdx >= 0; --stationIdx)
      if(theStationMask & 1<<stationIdx) {
         lastSegBit = stationIdx;
         break;
      }

   int station = 0, detector = 0;
   station  = lastSegBit < 4 ? lastSegBit+1 : lastSegBit-3;
   detector = lastSegBit < 4 ? 1 : 2;

   if(lastSegBit != 3) {
      if(fabs(muon.pullX(station,detector,arbitrationType,1)) > maxAbsPullX &&
	 fabs(muon.dX(station,detector,arbitrationType)) > maxAbsDx)
	goodMuon = false;
      if(fabs(muon.pullY(station,detector,arbitrationType,1)) > maxAbsPullY &&
	 fabs(muon.dY(station,detector,arbitrationType)) > maxAbsDy)
	goodMuon = false;
   } else {
      // special consideration for dt where there is no y information in station 4
      // impose y cuts on next station with segment
      if(fabs(muon.pullX(4,1,arbitrationType,1)) > maxAbsPullX &&
	 fabs(muon.dX(4,1,arbitrationType)) > maxAbsDx)
	goodMuon = false;
      if(theStationMask & 1<<2) {
	 if(fabs(muon.pullY(3,1,arbitrationType,1)) > maxAbsPullY &&
	    fabs(muon.dY(3,1,arbitrationType)) > maxAbsDy)
	   goodMuon = false;
      } else if(theStationMask & 1<<1) {
	 if(fabs(muon.pullY(2,1,arbitrationType,1)) > maxAbsPullY &&
	    fabs(muon.dY(2,1,arbitrationType)) > maxAbsDy)
	   goodMuon = false;
      } else if(theStationMask & 1<<0) {
	 if(fabs(muon.pullY(1,1,arbitrationType,1)) > maxAbsPullY &&
	    fabs(muon.dY(1,1,arbitrationType)) > maxAbsDy)
	   goodMuon = false;
      }
   }
   
   return goodMuon;
}

bool muon::isGoodMuon( const reco::Muon& muon, reco::Muon::SelectionType type )
{
  switch (type)
     {
      case reco::Muon::All:
	return true;
	break;
      case reco::Muon::AllGlobalMuons:
	return muon.isGlobalMuon();
	break;
      case reco::Muon::AllTrackerMuons:
	return muon.isTrackerMuon();
	break;
      case reco::Muon::AllStandAloneMuons:
	return muon.isStandAloneMuon();
	break;
      case reco::Muon::TrackerMuonArbitrated:
	return muon.isTrackerMuon() && muon.numberOfMatches(reco::Muon::SegmentAndTrackArbitration)>0;
	break;
      case reco::Muon::AllArbitrated:
	return ! muon.isTrackerMuon() || muon.numberOfMatches(reco::Muon::SegmentAndTrackArbitration)>0;
	break;
      case reco::Muon::GlobalMuonPromptTight:
	return 
	  muon.globalTrack()->normalizedChi2()<5 &&
	  fabs(muon.innerTrack()->d0()) < 0.25 &&
	  muon.innerTrack()->numberOfValidHits() >= 7;
	break;
      case reco::Muon::TMLastStationLoose:
	return isGoodMuon(muon,TMLastStation,2,3,3,9999,9999,-3,-3,reco::Muon::SegmentAndTrackArbitration);
	break;
      case reco::Muon::TMLastStationTight:
	return isGoodMuon(muon,TMLastStation,2,3,3,3,3,-3,-3,reco::Muon::SegmentAndTrackArbitration);
	break;
	//compatibility loose
      case reco::Muon::TM2DCompatibilityLoose:
	return isGoodMuon(muon,TM2DCompatibility,0.7);
	break;
	//compatibility tight
      case reco::Muon::TM2DCompatibilityTight:
	return isGoodMuon(muon,TM2DCompatibility,1.1);
	break;
      default:
	return false;
     }
}
