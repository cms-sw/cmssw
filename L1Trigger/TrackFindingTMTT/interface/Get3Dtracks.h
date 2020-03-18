#ifndef __Get3Dtracks_H__
#define __Get3Dtracks_H__

#include "L1Trigger/TrackFindingTMTT/interface/TrkRZfilter.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"

#include "boost/numeric/ublas/matrix.hpp"
#include <vector>
#include <utility>

using  boost::numeric::ublas::matrix;

using namespace std;

//=== This reconstructs 3D tracks from the 2D tracks found by the Hough transform.
//=== It can do this by simply estimating the r-z helix parameters from the centre of the eta sector
//=== and/or by running an r-z filter (e.g. Seed Filter), which also cleans up the tracks by
//=== checking their stubs consistency with a straight line in the r-z plane.
//=== 
//=== To create 3D tracks, call the sequence init(), run(), and then get tracks via trackCands3D().

namespace TMTT {

class Settings;
class Stub;
class TP;

class Get3Dtracks {

public:
  
  Get3Dtracks() : settings_(nullptr), iPhiSec_(0), iEtaReg_(0), etaMinSector_(0), etaMaxSector_(0), phiCentreSector_(0), runRZfilter_(false) {}
  ~Get3Dtracks() {}

  //=== Main routines to make 3D tracks.

  // Initialization
  void init(const Settings* settings, unsigned int iPhiSec, unsigned int iEtaReg, 
            float etaMinSector, float etaMaxSector, float phiCentreSector);

  // Make 3D track collections.
  void run(const vector<L1track2D>& vecTracksRphi) {
    this->makeUnfilteredTrks(vecTracksRphi);
    if (runRZfilter_) this->makeRZfilteredTrks(vecTracksRphi);
  }

  //=== Get 3D tracks.

  // Get 3D tracks (either r-z filtered or unfiltered, depending on the boolean).
  // (Each L1track3D object gives access to stubs on each track and helix parameters 
  // & also to the associated truth tracking particle).
  const vector<L1track3D>& trackCands3D(bool rzFiltered) const {
    if (rzFiltered) {
      return vecTracks3D_rzFiltered_;
    } else {
      return vecTracks3D_unfiltered_;
    }
  }

  // Get all 3D track candidates (either r-z filtered on unfiltered, depending on the boolean), 
  // that are associated to the given tracking particle.
  // (If the vector is empty, then the tracking particle was not reconstructed in this sector).
  vector<const L1track3D*> assocTrackCands3D(const TP& tp, bool rzFiltered) const;

  //=== Access to track r-z filter in case internal info from it required.

  bool ranRZfilter() const {return runRZfilter_;} // Was r-z filter required/run?

  const TrkRZfilter& getRZfilter() const {return rzFilter_;}

private:

  // Convert 2D HT tracks within the current sector to 3D tracks,
  // by adding a rough estimate of their r-z helix parameters, without running any r-z track filter.
  void makeUnfilteredTrks(const vector<L1track2D>& vecTracksRphi);

  // Make 3D tracks from the 2D HT tracks within the current sector, by running the r-z track filter.
  // The r-z filter also adds an estimate of the r-z helix parameters to each track.
  // (Not filled if no track fitter needs the r-z filter).  
  void makeRZfilteredTrks(const vector<L1track2D>& vecTracksRphi);

private:

  // Configuration parameters
  const Settings* settings_;
  unsigned int iPhiSec_;   // Sector number.
  unsigned int iEtaReg_;
  float etaMinSector_;     // Range of eta sector
  float etaMaxSector_;     // Range of eta sector
  float phiCentreSector_;  // Phi angle of centre of this (eta,phi) sector.

  bool runRZfilter_;       // Does r-z track filter need to be run.

  // Track filter(s), such as r-z filters, run after the r-phi Hough transform.
  TrkRZfilter rzFilter_;

  // List of all found 3D track candidates and their associated properties.
  vector<L1track3D> vecTracks3D_rzFiltered_; // After r-z filter run
  vector<L1track3D> vecTracks3D_unfiltered_; // Before r-z filter run.
};

}
#endif

