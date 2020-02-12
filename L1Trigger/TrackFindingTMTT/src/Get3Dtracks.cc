#include "L1Trigger/TrackFindingTMTT/interface/Get3Dtracks.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track2D.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"

#include <iostream>
#include <unordered_set>

using namespace std;

namespace TMTT {

class Settings;

//=== Initialization

void Get3Dtracks::init(const Settings* settings, unsigned int iPhiSec, unsigned int iEtaReg,
		  float etaMinSector, float etaMaxSector, float phiCentreSector) {

  // Store config params & arguments.
  settings_        = settings;
  iPhiSec_         = iPhiSec;                  // Sector number
  iEtaReg_         = iEtaReg;
  etaMinSector_    = etaMinSector;             // Range of eta sector
  etaMaxSector_    = etaMaxSector;             // Range of eta sector
  phiCentreSector_ = phiCentreSector;          // Centre of phi sector

  // Note if any fitters require an r-z track filter to be run.
  runRZfilter_     = (settings->useRZfilter().size() > 0);

  // Initialize any track filters (e.g. r-z) run after the r-phi Hough transform.
  if (runRZfilter_) rzFilter_.init(settings_, iPhiSec_, iEtaReg_, etaMinSector_, etaMaxSector_, phiCentreSector_);  
}

//=== Convert 2D tracks found by HT within the current sector to 3D tracks, without running any r-z track filter.
//=== by adding a rough estimate of their r-z helix parameters.

void Get3Dtracks::makeUnfilteredTrks(const vector<L1track2D>& vecTracksRphi) {

  vecTracks3D_unfiltered_.clear();

  for (const L1track2D& trkRphi : vecTracksRphi) {
    const vector<const Stub*>& stubsOnTrkRphi = trkRphi.getStubs(); // stubs assigned to track 

    float qOverPt = trkRphi.getHelix2D().first;
    float phi0    = trkRphi.getHelix2D().second;

    if (settings_->enableDigitize()) {
      // Centre of HT bin lies on boundary of two fitted track digi bins, so nudge slightly +ve (like FW)
      // to remove ambiguity.
      const float small = 0.1;
      const unsigned int nHelixBits = 18; // Bits used internally in KF HLS to represent helix params.
      qOverPt += (2./settings_->invPtToInvR()) *
                 small*settings_->kf_oneOver2rRange()/pow(2., nHelixBits);
      phi0    += small*settings_->kf_phi0Range()     /pow(2., nHelixBits);
    }    
    pair<float, float> helixRphi(qOverPt, phi0);

    // Estimate r-z track helix parameters from centre of eta sector.
    float z0 = 0.;
    float tan_lambda = 0.5*(1/tan(2*atan(exp(-etaMinSector_))) + 1/tan(2*atan(exp(-etaMaxSector_))));

    // float etaCentreSector = 0.5*(etaMinSector_ + etaMaxSector_);
    // float theta = 2. * atan(exp(-etaCentreSector));
    // tan_lambda = 1./tan(theta);

    pair<float, float> helixRz(z0, tan_lambda);

    // Create 3D track, by adding r-z helix params to 2D track
    L1track3D trk3D(settings_, stubsOnTrkRphi, 
		    trkRphi.getCellLocationHT(), helixRphi, helixRz,
		    iPhiSec_                   , iEtaReg_            , trkRphi.optoLinkID(), trkRphi.mergedHTcell());
    //    L1track3D trk3D(settings_, stubsOnTrkRphi, 
    //		    trkRphi.getCellLocationHT(), trkRphi.getHelix2D(), helixRz,
    //		    iPhiSec_                   , iEtaReg_            , trkRphi.optoLinkID(), trkRphi.mergedHTcell());

    // Optionally use MC truth to eliminate all fake tracks & all incorrect stubs assigned to tracks 
    // before doing fit (for debugging).
    bool cheat_keep = true;
    if (settings_->trackFitCheat()) cheat_keep = trk3D.cheat();

    // Add to list of stored 3D tracks.
    if (cheat_keep) vecTracks3D_unfiltered_.push_back( trk3D );
  }
}

//=== Make 3D tracks from the 2D tracks found by the HT within the current sector, by running the r-z track filter.
//=== The r-z filter also adds an estimate of the r-z helix parameters to each track.

void Get3Dtracks::makeRZfilteredTrks(const vector<L1track2D>& vecTracksRphi) {
  vecTracks3D_rzFiltered_ = rzFilter_.filterTracks(vecTracksRphi);

  // Optionally use MC truth to eliminate all fake tracks & all incorrect stubs assigned to tracks 
  // before doing fit (for debugging).
  if (settings_->trackFitCheat()) {
    vector<L1track3D> vecTracks3D_tmp;
    for (const L1track3D& trk : vecTracks3D_rzFiltered_) {
      L1track3D trk_tmp = trk;
      bool cheat_keep = trk_tmp.cheat();
      if (cheat_keep) vecTracks3D_tmp.push_back(trk_tmp);
    }
    vecTracks3D_rzFiltered_ = vecTracks3D_tmp;
  }
}

//=== Get all 3D track candidates (either r-z filtered on unfiltered, depending on the boolean), 
//=== that are associated to the given tracking particle.
//=== (If the vector is empty, then the tracking particle was not reconstructed in this sector).

vector<const L1track3D*> Get3Dtracks::assocTrackCands3D(const TP& tp, bool rzFiltered) const {

  const vector<L1track3D>& allTracks3D = (rzFiltered)  ?  vecTracks3D_rzFiltered_  :  vecTracks3D_unfiltered_;

  vector<const L1track3D*> assocRecoTrk;

  // Loop over track candidates, looking for those associated to given TP.
  for (const L1track3D& trk : allTracks3D) {
    if (trk.getMatchedTP() != nullptr) {
      if (trk.getMatchedTP()->index() == tp.index()) assocRecoTrk.push_back(&trk); 
    }
  }

  return assocRecoTrk;
}

}
