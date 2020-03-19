#include "L1Trigger/TrackFindingTMTT/interface/HTbase.h"
#include "L1Trigger/TrackFindingTMTT/interface/InputData.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <vector>
#include <unordered_set>

using namespace std;

namespace TMTT {

//=== Termination. Causes HT array to search for tracks etc.

void HTbase::end() {

  // Calculate useful info about each cell in array.
  for (unsigned int i = 0; i < htArray_.size1(); i++) {
    for (unsigned int j = 0; j < htArray_.size2(); j++) {
      htArray_(i,j).end(); // Calls HTcell::end()
    }
  }

  // Produce a list of all track candidates found in this array, each containing all the stubs on each one
  // and the track helix parameters, plus the associated truth particle (if any).
  trackCands2D_ = this->calcTrackCands2D();

  // Run algorithm to kill duplicate tracks (e.g. those sharing many hits in common).
  trackCands2D_ = killDupTrks_.filter( trackCands2D_ );

  // If requested, kill those tracks in this sector that can't be read out during the time-multiplexed period, because
  // the HT has associated too many stubs to tracks. 
  if (settings_->busySectorKill()) {
    trackCands2D_ = this->killTracksBusySec( trackCands2D_ );
  }
}

//=== Number of filtered stubs in each cell summed over all cells in HT array.
//=== If a stub appears in multiple cells, it will be counted multiple times.
unsigned int HTbase::numStubsInc() const {

  unsigned int nStubs = 0;

  // Loop over cells in HT array.
  for (unsigned int i = 0; i < htArray_.size1(); i++) {
    for (unsigned int j = 0; j < htArray_.size2(); j++) {
      nStubs += htArray_(i,j).numStubs(); // Calls HTcell::numStubs()
    }
  }

  return nStubs;
}

//=== Number of filtered stubs in HT array.
//=== If a stub appears in multiple cells, it will be counted only once.
unsigned int HTbase::numStubsExc() const {

  unordered_set<unsigned int> stubIDs; // Each ID stored only once, no matter how often it is added.

  // Loop over cells in HT array.
  for (unsigned int i = 0; i < htArray_.size1(); i++) {
    for (unsigned int j = 0; j < htArray_.size2(); j++) {
      // Loop over stubs in each cells, storing their IDs.
      const vector<const Stub*>& vStubs = htArray_(i,j).stubs(); // Calls HTcell::stubs()
      for (const Stub* stub : vStubs) {
        stubIDs.insert( stub->index() );
      }
    }
  }

  return stubIDs.size();
}

//=== Get number of filtered stubs assigned to track candidates found in this HT array.

unsigned int HTbase::numStubsOnTrackCands2D() const {

  unsigned int nStubs = 0;

  // Loop over track candidates
  for (const L1track2D& trk : trackCands2D_) {
    nStubs += trk.getStubs().size();
  }

  return nStubs;
}

//=== Get all reconstructed tracks that were associated to the given tracking particle.
//=== (If the vector is empty, then the tracking particle was not reconstructed in this sector).

vector<const L1track2D*> HTbase::assocTrackCands2D(const TP& tp) const {

  vector<const L1track2D*> assocRecoTrk;

  // Loop over track candidates, looking for those associated to given TP.
  for (const L1track2D& trk : trackCands2D_) {
    if (trk.getMatchedTP() != nullptr) {
      if (trk.getMatchedTP()->index() == tp.index()) assocRecoTrk.push_back(&trk);
    } 
  }

  return assocRecoTrk;
}

//=== Function to replace the collection of 2D tracks found by this HT.
//=== (This is used by class MuxHToutputs to kill tracks that can't be output in the time-multiplexed period).

void HTbase::replaceTrackCands2D(const vector<const L1track2D*>& newTracks) {
  vector<L1track2D> tmpTracks;
  for (const L1track2D* trk : newTracks) {
    tmpTracks.push_back(*trk);
  }
  trackCands2D_.clear();
  trackCands2D_ = tmpTracks;
}

//=== Disable filters (used for debugging).

void HTbase::disableBendFilter() {
  // Loop over cells in HT array.
  for (unsigned int i = 0; i < htArray_.size1(); i++) {
    for (unsigned int j = 0; j < htArray_.size2(); j++) {
      htArray_(i,j).disableBendFilter();
    }
  }
}

//=== Given a range in one of the coordinates specified by coordRange, calculate the corresponding range of bins. The other arguments specify the axis. And also if some cells nominally associated to stub are to be killed.

pair<unsigned int, unsigned int> HTbase::convertCoordRangeToBinRange( pair<float, float> coordRange, unsigned int nBinsAxis, float coordAxisMin, float coordAxisBinSize, unsigned int killSomeHTcells, bool debug) const {

  float coordMin = coordRange.first;
  float coordMax = coordRange.second;
  float coordAvg = ( coordRange.first + coordRange.second ) / 2.;

  int iCoordBinMin, iCoordBinMax;

  //--- There are various options for doing this.
  //--- Option killSomeHTcells = 0 is the obvious one.
  //--- If killSomeHTcells > 0, then some of the cells nominally associated with the stub are killed.

  if (killSomeHTcells == 0) {
    // Take the full range of phi bins consistent with the stub.
    iCoordBinMin = floor( ( coordMin - coordAxisMin ) / coordAxisBinSize );
    iCoordBinMax = floor( ( coordMax - coordAxisMin ) / coordAxisBinSize );
  } else if (killSomeHTcells == 1) {
    // Use the reduced range of bins.
    // This algorithm, proposed by Ian, should reduce the rate, at the cost of some efficiency.
    const float fracCut = 0.3;
    iCoordBinMin = floor( (  coordMin - coordAxisMin ) / coordAxisBinSize );
    iCoordBinMax = floor( (  coordMax - coordAxisMin ) / coordAxisBinSize );
    unsigned int nbins = iCoordBinMax - iCoordBinMin + 1;
    if (nbins >= 2) { // Can't reduce range if already only 1 bin 
      float lower = coordAxisMin + (iCoordBinMin + 1) * coordAxisBinSize; // upper edge of lowest bin
      float upper = coordAxisMin + (iCoordBinMax    ) * coordAxisBinSize; // lower edge of highest bin.
      // Calculate fractional amount of min and max bin that this stub uses.
      float extraLow = (lower - coordMin) / coordAxisBinSize;
      float extraUp  = (coordMax - upper) / coordAxisBinSize; 
      if (min(extraLow,extraUp) < -0.001 || max(extraLow, extraUp) > 1.001) cout<<"THIS SHOULD NOT HAPPEN "<<extraLow<<endl; // allowing 0.001 tolerance for floating point precision here.
      if (extraLow < fracCut && (nbins >= 3 || extraLow < extraUp)) iCoordBinMin += 1;
      if (extraUp  < fracCut && (nbins >= 3 || extraUp < extraLow)) iCoordBinMax -= 1;       
    }
  } else if ( killSomeHTcells == 2 ) {
    // This corresponds to Thomas's firmware implementation, which can't fill more than one HT cell per column.
    iCoordBinMin = floor( ( coordAvg - coordAxisMin ) / coordAxisBinSize );
    iCoordBinMax = iCoordBinMin;
  } else {
    throw cms::Exception("HT: invalid HoughUseFullRange option in cfg");
  }

  if (debug) cout<<"Initial Coord range: "<<coordMin<<" "<<coordMax<<" "<<iCoordBinMin<<" "<<iCoordBinMax<<endl;

  // Limit range to dimensions of HT array.
  iCoordBinMin = max(iCoordBinMin, 0);
  iCoordBinMax = min(iCoordBinMax, int(nBinsAxis) - 1);

  if (debug) {
    float downPhiLim = coordAxisMin +  iCoordBinMin * coordAxisBinSize;
    float upPhiLim   = coordAxisMin + (iCoordBinMax + 1) * coordAxisBinSize;
    cout<<"Final Coord range: "<<downPhiLim<<" "<<upPhiLim<<" "<<iCoordBinMin<<" "<<iCoordBinMax<<endl;
  }

  // If whole range is outside HT array, flag this by setting range to specific values with min > max.
  if (iCoordBinMin > int(nBinsAxis) - 1 || iCoordBinMax < 0) {
    iCoordBinMin = int(nBinsAxis) - 1;
    iCoordBinMax = 0;
  }

  return pair<unsigned int, unsigned int>(iCoordBinMin, iCoordBinMax);
} 

//=== Return a list of all track candidates found in this array, giving access to all the stubs on each one
//=== and the track helix parameters, plus the associated truth particle (if any).

vector<L1track2D> HTbase::calcTrackCands2D() const {

  vector<L1track2D> trackCands2D;

  if (settings_->debug() == 3) cout<<"Printing track candidates in an HT array"<<endl;

  const unsigned int numRows = htArray_.size1();
  const unsigned int numCols = htArray_.size2();

  // Check if the hardware processes rows of the HT array in a specific order when outputting track candidates.
  // Currently this is by decreasing Pt for r-phi HT and unordered for r-z HT.
  const vector<unsigned int> iOrder = this->rowOrder(numRows);
  bool wantOrdering = (iOrder.size() > 0);

  unsigned int numStubsLeft = 0;
  unsigned int numStubsRight = 0;

  // Loop over cells in HT array.
  for (unsigned int i = 0; i < numRows; i++) {

    // Access rows in specific order if required.
    unsigned int iPos = wantOrdering  ?   iOrder[i]  :  i;

    for (unsigned int j = 0; j < numCols; j++) {
      if (htArray_(iPos,j).trackCandFound()) { // track candidate found in this cell.

	// Note if this corresponds to a merged HT cell (e.g. 2x2).
        const bool merged = htArray_(iPos,j).mergedCell();

	// Get stubs on this track candidate.
        const vector<const Stub*>& stubs = htArray_(iPos,j).stubs();

	// And note location of cell inside HT array.
        const pair<unsigned int, unsigned int> cellLocation(iPos, j);

        // Get (q/Pt, phi0) or (tan_lambda, z0) corresponding to middle of this cell.
        const pair<float, float> helixParams2D = this->helix2Dconventional(iPos, j);

	// Store all this reconstruction info about this track.
	// The L1track2D class automatically finds the associated MC truth Tracking Particle particle (if any)
        L1track2D l1Trk2D(settings_, stubs, cellLocation, helixParams2D, iPhiSec_, iEtaReg_, optoLinkID_, merged);

	// Store all this info about the track.        
	trackCands2D.push_back( l1Trk2D );

      } else {
	if (settings_->debug() == 3) cout<<" ."; // Indicate no track in this cell.
      }
    }
    if (settings_->debug() == 3) cout<<endl;
  }
  
  return trackCands2D;
}


}
