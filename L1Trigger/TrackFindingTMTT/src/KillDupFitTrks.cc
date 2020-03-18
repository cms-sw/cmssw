#include "L1Trigger/TrackFindingTMTT/interface/KillDupFitTrks.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <map>

namespace TMTT {

//=== Make available cfg parameters & specify which algorithm is to be used for duplicate track removal.

void KillDupFitTrks::init(const Settings* settings, unsigned int dupTrkAlg)
{
  settings_ = settings;
  dupTrkAlg_ = dupTrkAlg;
  killDupTrks_.init(settings, dupTrkAlg); // Initialise duplicate removal algorithms that are common to all tracks.
}

//=== Eliminate duplicate tracks from the input collection, and so return a reduced list of tracks.

vector<L1fittedTrack> KillDupFitTrks::filter(const vector<L1fittedTrack>& vecTracks) const
{
  if (dupTrkAlg_ == 0) {

    // We are not running duplicate removal, so return original fitted track collection.    
    return vecTracks;

  } else {

    // Choose which algorithm to run, based on parameter dupTrkAlg_.
    switch (dupTrkAlg_) {
      // Run filters that only work on fitted tracks.
      case 50:  return filterAlg50( vecTracks )  ; break;
      case 51:  return filterAlg51( vecTracks ); break;
      // Run filters that work on any type of track (l1track2d, l1track3d, l1fittedtrack). 
      default:  return killDupTrks_.filter(vecTracks); 
    }

    // Should never get here ...	
  }
}

//=== Duplicate removal algorithm designed to run after the track helix fit, which eliminates duplicates  
//=== simply by requiring that the fitted (q/Pt, phi0) of the track correspond to the same HT cell in 
//=== which the track was originally found by the HT.
//=== N.B. This code runs on tracks in a single sector. It could be extended to run on tracks in entire
//=== tracker by adding the track's sector number to memory "htCellUsed" below.


vector<L1fittedTrack> KillDupFitTrks::filterAlg50(const vector<L1fittedTrack>& tracks) const
{
  // Hard-wired options to play with.
  const bool debug = false;
  const bool doRecoveryStep = true; // Do 2nd pass through rejected tracks to see if any should be rescued.
  const bool reduceDups = true; // Option attempting to reduce duplicate tracks during 2nd pass.
  const bool memorizeAllHTcells = false; // First pass stores in memory all cells that the HT found tracks in, not just those of tracks accepted by the first pass.
  const bool doSectorCheck = false; // Require fitted helix to lie within sector.
  const bool usePtAndZ0Cuts = false;
  // IRT - was false
  const bool goOutsideArray = true; // Also store in memory stubs outside the HT array during 2nd pass.
  // IRT  - was false
  const bool limitDiff = true; // Limit allowed diff. between HT & Fit cell to <= 1.

  if (debug && tracks.size() > 0) cout<<"START "<<tracks.size()<<endl;

  vector<L1fittedTrack> tracksFiltered;

  // Make a first pass through the tracks, doing initial identification of duplicate tracks.
  // N.B. BY FILLING THIS WITH CELLS AROUND SELECTED TRACKS, RATHER THAN JUST THE CELL CONTAINING THE
  // TRACK, ONE CAN REDUCE THE DUPLICATE RATE FURTHER, AT COST TO EFFICIENCY.
  set< pair<unsigned int, unsigned int> > htCellUsed;
  vector<const L1fittedTrack*> tracksRejected;

  // For checking if multiple tracks corresponding to same TP are accepted by duplicate removal.
  map<unsigned int, pair<unsigned int, unsigned int>> tpFound;
  map<unsigned int, unsigned int>                     tpFoundAtPass;

  for (const L1fittedTrack& trk : tracks) {

    // Only consider tracks whose fitted helix parameters are in the same sector as the HT originally used to find the track.
    if ( ( ! doSectorCheck) || trk.consistentSector() ) {
      if ( ( ! usePtAndZ0Cuts) || ( fabs(trk.z0()) < settings_->beamWindowZ() && trk.pt() > settings_->houghMinPt() - 0.2) ) {
	
    // For debugging.
    const TP* tp = trk.getMatchedTP();

  // Check if this track's fitted (q/pt, phi0) helix parameters correspond to the same HT cell as the HT originally found the track in.
	bool consistentCell = trk.consistentHTcell();
	if (consistentCell) {
	  // This track is probably not a duplicate, so keep & and store its HT cell location (which equals the HT cell corresponding to the fitted track).
	  tracksFiltered.push_back(trk);
	  // Memorize HT cell location corresponding to this track (identical for HT track & fitted track).
	  if ( ! memorizeAllHTcells) {
	    pair<unsigned int, unsigned int> htCell = trk.getCellLocationHT();
	    htCellUsed.insert( htCell );
   	    if (trk.getL1track3D().mergedHTcell()) {
	      // If this is a merged cell, block the other elements too, in case a track found by the HT in an unmerged cell
	      // has a fitted cell there.
	      pair<unsigned int, unsigned int> htCell10( htCell.first + 1, htCell.second);
	      pair<unsigned int, unsigned int> htCell01( htCell.first    , htCell.second + 1);
	      pair<unsigned int, unsigned int> htCell11( htCell.first + 1, htCell.second + 1);
	      htCellUsed.insert( htCell10 );
	      htCellUsed.insert( htCell01 );
	      htCellUsed.insert( htCell11 );
	    }
	  }

	  if (debug && tp != nullptr) {
	    cout<<"FIRST PASS: m="<<trk.getCellLocationHT().first<<"/"<<trk.getCellLocationFit().first<<" c="<<trk.getCellLocationHT().second<<"/"<<trk.getCellLocationFit().second<<" Delta(m,c)=("<<int(trk.getCellLocationHT().first) - int(trk.getCellLocationFit().first)<<","<<int(trk.getCellLocationHT().second) - int(trk.getCellLocationFit().second)<<") pure="<<trk.getPurity()<<" merged="<<trk.getL1track3D().mergedHTcell()<<" #layers="<<trk.getL1track3D().getNumLayers()<<" tp="<<tp->index()<<" dupCell=("<<tpFound[tp->index()].first<<","<<tpFound[tp->index()].second<<") dup="<<tpFoundAtPass[tp->index()]<<endl;
	    // If the following two variables are non-zero in printout, then track has already been found,  
	    // so we have mistakenly kept a duplicate.
	    if (tpFound.find(tp->index()) != tpFound.end()) tpFound[tp->index()] = trk.getCellLocationFit();
	    tpFoundAtPass[tp->index()] = 1;
	  }

	} else {

	  if (limitDiff) {
	    const unsigned int maxDiff = 1;
  	    if (abs(int(trk.getCellLocationHT().first)  - int(trk.getCellLocationFit().first))  <= maxDiff &&
	        abs(int(trk.getCellLocationHT().second) - int(trk.getCellLocationFit().second)) <= maxDiff) tracksRejected.push_back(&trk);
	  } else {
	    tracksRejected.push_back(&trk);
          }

	  if (debug && tp != nullptr) {
            cout<<"FIRST REJECT: m="<<trk.getCellLocationHT().first<<"/"<<trk.getCellLocationFit().first<<" c="<<trk.getCellLocationHT().second<<"/"<<trk.getCellLocationFit().second<<" Delta(m,c)=("<<int(trk.getCellLocationHT().first) - int(trk.getCellLocationFit().first)<<","<<int(trk.getCellLocationHT().second) - int(trk.getCellLocationFit().second)<<") pure="<<trk.getPurity()<<" merged="<<trk.getL1track3D().mergedHTcell()<<" #layers="<<trk.getL1track3D().getNumLayers()<<" tp="<<tp->index()<<" dupCell=("<<tpFound[tp->index()].first<<","<<tpFound[tp->index()].second<<") dup="<<tpFoundAtPass[tp->index()]<<endl;
	  }
	}
	// Memorize HT cell location corresponding to this track, even if it was not accepted by first pass..
	if (memorizeAllHTcells) {
	  pair<unsigned int, unsigned int> htCell = trk.getCellLocationFit(); // Intentionally used fit instead of HT here.
	  htCellUsed.insert( htCell );
	  if (trk.getL1track3D().mergedHTcell()) {
	    // If this is a merged cell, block the other elements too, in case a track found by the HT in an unmerged cell
	    // has a fitted cell there.
	    // N.B. NO GOOD REASON WHY "-1" IS NOT DONE HERE TOO. MIGHT REDUCE DUPLICATE RATE?
	    pair<unsigned int, unsigned int> htCell10( htCell.first + 1, htCell.second);
	    pair<unsigned int, unsigned int> htCell01( htCell.first    , htCell.second + 1);
	    pair<unsigned int, unsigned int> htCell11( htCell.first + 1, htCell.second + 1);
	    htCellUsed.insert( htCell10 );
	    htCellUsed.insert( htCell01 );
	    htCellUsed.insert( htCell11 );
	  }
	}
      }
    }
  }

  if (doRecoveryStep) {
    // Making a second pass through the rejected tracks, checking if any should be rescued.
    for (const L1fittedTrack* trk : tracksRejected) {


      // Get location in HT array corresponding to fitted track helix parameters.
      pair<unsigned int, unsigned int> htCell = trk->getCellLocationFit();
      // If this HT cell was not already memorized, rescue this track, since it is probably not a duplicate,
      // but just a track whose fitted helix parameters are a bit wierd for some reason.
      if (std::count(htCellUsed.begin(), htCellUsed.end(), htCell) == 0) {
	tracksFiltered.push_back(*trk); // Rescue track.
	// Optionally store cell location to avoid rescuing other tracks at the same location, which may be duplicates of this track. 
	bool outsideCheck =( goOutsideArray || trk->pt() > settings_->houghMinPt() );
	if (reduceDups && outsideCheck) htCellUsed.insert( htCell );

  // For debugging.
  const TP* tp = trk->getMatchedTP();

	if (debug && tp != nullptr) {
	  cout<<"SECOND PASS: m="<<trk->getCellLocationHT().first<<"/"<<trk->getCellLocationFit().first<<" c="<<trk->getCellLocationHT().second<<"/"<<trk->getCellLocationFit().second<<" Delta(m,c)=("<<int(trk->getCellLocationHT().first) - int(trk->getCellLocationFit().first)<<","<<int(trk->getCellLocationHT().second) - int(trk->getCellLocationFit().second)<<") pure="<<trk->getPurity()<<" merged="<<trk->getL1track3D().mergedHTcell()<<" #layers="<<trk->getL1track3D().getNumLayers()<<" tp="<<tp->index()<<" dupCell=("<<tpFound[tp->index()].first<<","<<tpFound[tp->index()].second<<") dup="<<tpFoundAtPass[tp->index()]<<endl;
	  if (tpFound.find(tp->index()) != tpFound.end()) tpFound[tp->index()] = htCell;
	  tpFoundAtPass[tp->index()] = 2;
	}
      }
    }
  }

  // Debug printout to identify duplicate tracks that survived.
  if (debug) this->printDuplicateTracks(tracksFiltered);

  return tracksFiltered;
}
//=== Duplicate removal algorithm designed to run after the track helix fit, which eliminates duplicates  
//=== simply by requiring that no two tracks should have fitted (q/Pt, phi0) that correspond to the same HT
//=== cell. If they do, then only the first to arrive is kept.
//=== N.B. This code runs on tracks in a single sector. It could be extended to run on tracks in entire
//=== tracker by adding the track's sector number to memory "htCellUsed" below.

vector<L1fittedTrack> KillDupFitTrks::filterAlg51(const vector<L1fittedTrack>& tracks) const
{
  // Hard-wired options to play with.
  const bool debug = false;

  if (debug && tracks.size() > 0) cout<<"START "<<tracks.size()<<endl;

  vector<L1fittedTrack> tracksFiltered;
  set< pair<unsigned int, unsigned int> > htCellUsed;

  for (const L1fittedTrack& trk : tracks) {
      // Get location in HT array corresponding to fitted track helix parameters.
      pair<unsigned int, unsigned int> htCell = trk.getCellLocationFit();
      // If this HT cell was not already memorized, rescue this track, since it is probably not a duplicate,
      // but just a track whose fitted helix parameters are a bit wierd for some reason.
      if (std::count(htCellUsed.begin(), htCellUsed.end(), htCell) == 0) {
	tracksFiltered.push_back(trk); // Rescue track.
	// Store cell location to avoid rescuing other tracks at the same location, which may be duplicates of this track. 
	htCellUsed.insert( htCell );
	if (debug) {
	  const TP* tp = trk.getMatchedTP();
	  int tpIndex = (tp != nullptr) ? tp->index() : -999;
	  cout<<"ALG51: m="<<trk.getCellLocationHT().first<<"/"<<trk.getCellLocationFit().first<<" c="<<trk.getCellLocationHT().second<<"/"<<trk.getCellLocationFit().second<<" tp="<<tpIndex<<" pure="<<trk.getPurity()<<endl;
	}
      }
    }

  // Debug printout to identify duplicate tracks that survived.
  if (debug) this->printDuplicateTracks(tracksFiltered);

  return tracksFiltered;
}

// Debug printout of which tracks are duplicates.
void KillDupFitTrks::printDuplicateTracks(const vector<L1fittedTrack>& tracks) const {
  map<const TP*, vector<const L1fittedTrack*> > tpMap;
  for (const L1fittedTrack& trk : tracks) {
    const TP* tp = trk.getMatchedTP();
    if (tp != nullptr) {
      tpMap[tp].push_back(&trk);
    }
  }
  for (const auto& p : tpMap) {
    const TP* tp     = p.first;
    const vector<const L1fittedTrack*> vecTrk = p.second;
    if (vecTrk.size() > 1) {
      for (const L1fittedTrack* trk : vecTrk) {
	cout<<"  MESS UP : m="<<trk->getCellLocationHT().first<<"/"<<trk->getCellLocationFit().first<<" c="<<trk->getCellLocationHT().second<<"/"<<trk->getCellLocationFit().second<<" tp="<<tp->index()<<" tp_pt="<<tp->pt()<<" fit_pt="<<trk->pt()<<" pure="<<trk->getPurity()<<endl;
	cout<<"     stubs = ";
	for (const Stub* s : trk->getStubs()) cout<<s->index()<<" ";
	cout<<endl;
      }
    }
  }
  if (tracks.size() > 0) cout<<"FOUND "<<tracks.size()<<endl;
}

}
