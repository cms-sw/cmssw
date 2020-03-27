#ifndef __HTbase_H__
#define __HTbase_H__

#include "L1Trigger/TrackFindingTMTT/interface/HTcell.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track2D.h"
#include "L1Trigger/TrackFindingTMTT/interface/KillDupTrks.h"

#include "boost/numeric/ublas/matrix.hpp"
#include <vector>
#include <utility>

using  boost::numeric::ublas::matrix;

using namespace std;


//=== Base class for Hough Transform array for a single (eta,phi) sector.

namespace TMTT {

class Settings;
class Stub;
class TP;
class L1fittedTrack;

class HTbase {

public:
  
  HTbase() {}
  virtual ~HTbase(){}

  // Initialization.
  virtual void init(const Settings* settings, unsigned int iPhiSec, unsigned int iEtaReg, 
		    float etaMinSector, float etaMaxSector, float) = 0;

  void init(const Settings* settings, unsigned int iPhiSec, unsigned int iEtaReg) {settings_ = settings; iPhiSec_ = iPhiSec; iEtaReg_ = iEtaReg; optoLinkID_ = this->calcOptoLinkID();}

  // Add stub to HT array.
  // N.B. The argument lists for this are different for r-phi & r-z HT, so unfortunately it can't be declared in base class.
  //virtual void store(const Stub*) = 0;

  // Termination. Causes HT array to search for tracks etc.
  virtual void end();

  //=== Get info about filtered stubs in HT array.
  //=== (N.B. The `filtered stubs' are stubs passing any requested stub filters, e.g. bend and/or rapidity. 
  //=== If no filters were requested, they are identical to the unfiltered stubs.)

  // Get sum of number of filtered stubs stored in each cell of HT array (so a stub appearing in multiple cells is counted multiple times).
  virtual unsigned int numStubsInc() const;

  // Get sum the number of filtered stubs in the HT array, where each individual stub is counted only once, even if it appears in multiple cells.
  virtual unsigned int numStubsExc() const;

  // Get all the cells that make up the array, which in turn give access to the stubs inside them.
  // N.B. You can use getAllCells().size1() and getAllCells().size2() to get the dimensions ofthe array.
  virtual const matrix<HTcell>& getAllCells() const {return htArray_;}

  //=== Info about track candidates found.

  // N.B. If a duplicate track filter was run inside the HT, this will contain the reduced list of tracks passing this filter.
  // N.B. If some tracks could not be read out during the TM period, then such tracks are deleted from this list.

  // Get list of all track candidates found in this HT array, giving access to stubs on each track
  // and helix parameters.
  virtual const vector<L1track2D>& trackCands2D() const {return trackCands2D_;}

  // Number of track candidates found in this HT array.
  // If a duplicate track filter was run, this will contain the reduced list of tracks passing this filter.
  virtual unsigned int numTrackCands2D() const {return trackCands2D_.size();}

  // Get number of filtered stubs assigned to track candidates found in this HT array.
  virtual unsigned int   numStubsOnTrackCands2D() const;

  // Get all reconstructed tracks that were associated to the given tracking particle.
  // (If the vector is empty, then the tracking particle was not reconstructed in this sector).
  virtual vector<const L1track2D*> assocTrackCands2D(const TP& tp) const;

  //=== Function to replace the collection of 2D tracks found by this HT.

  // (This is used by classes MuxHToutputs & MiniHTstage).
  virtual void replaceTrackCands2D(const vector<const L1track2D*>& newTracks);
  virtual void replaceTrackCands2D(const vector<L1track2D>& newTracks) {trackCands2D_ = newTracks;}

  //=== Utilities

  // Get the values of the track helix params corresponding to middle of a specified HT cell (i,j).
  // The helix parameters returned will be those corresponding to the two axes of the HT array.
  // So they might be (q/pt, phi65), (eta, z0) or (z110, z0) etc. depending on the configuration.
  virtual pair<float, float> helix2Dhough       (unsigned int i, unsigned int j) const = 0;

  // Get the values of the track helix params corresponding to middle of a specified HT cell (i,j).
  // The helix parameters returned will be always be (q/Pt, phi0) or (tan_lambda, z0), irrespective of 
  // how the axes of the HT array are defined.
  virtual pair<float, float> helix2Dconventional(unsigned int i, unsigned int j) const = 0;

  // Which cell in HT array should this TP be in, based on its true trajectory?
  // Returns 999999 in at least one index if TP not expected to be in any cell in this array.
  virtual pair<unsigned int, unsigned int> trueCell( const TP* tp ) const = 0;

  // Which cell in HT array should this fitted track be in, based on its fitted trajectory?
  // Returns 999999 in at least one index if fitted track not expected to be in any cell in this array.
  virtual pair<unsigned int, unsigned int> getCell( const L1fittedTrack* fitTrk) const = 0;

  // Disable filters (used for debugging).
  virtual void disableBendFilter();

protected:

  // Given a range in one of the coordinates specified by coordRange, calculate the corresponding range of bins. The other arguments specify the axis. And also if some cells nominally associated to stub are to be killed.
  virtual pair<unsigned int, unsigned int> convertCoordRangeToBinRange( pair<float, float> coordRange, unsigned int nBinsAxis, float coordAxisMin, float coordAxisBinSize, unsigned int killSomeHTcells, bool debug = false) const;

private:

  // Return a list of all track candidates found in this array, giving access to all the stubs on each one
  // and the track helix parameters, plus the associated truth particle (if any).
  virtual vector<L1track2D> calcTrackCands2D() const;

  // If requested, kill those tracks in this sector that can't be read out during the time-multiplexed period, because 
  // the HT has associated too many stubs to tracks.
  virtual vector<L1track2D> killTracksBusySec(const vector<L1track2D>& tracks) const = 0;

  // Define the order in which the hardware processes rows of the HT array when it outputs track candidates.
  virtual vector<unsigned int> rowOrder(unsigned int numRows) const = 0;

  // Calculate output opto-link ID from HT, assuming there is no MUX stage.
  virtual unsigned int calcOptoLinkID() const {
    unsigned int numPhiSecPerNon =  settings_->numPhiSectors() / settings_->numPhiNonants();
    return (iEtaReg_ * numPhiSecPerNon + iPhiSec_);
  }

protected:

  const Settings* settings_; // configuration parameters.

  unsigned int iPhiSec_; // Sector number.
  unsigned int iEtaReg_;

  unsigned int optoLinkID_; // ID of opto-link from HT to Track Fitter.

  // Hough transform array.
  // This has two dimensions, representing the two track helix parameters being varied.
  matrix<HTcell> htArray_; 

  // Contains algorithm used for duplicate track removal.
  KillDupTrks<L1track2D> killDupTrks_;

  // List of all track candidates found by HT & their associated properties.
  // If a duplicate track filter was run inside the HT, this will contain the reduced list of tracks passing this filter.
  // If some tracks could not be read out during the TM period, then such tracks are deleted from this list.
  vector<L1track2D> trackCands2D_;
};

}

#endif

