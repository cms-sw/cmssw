#include "L1Trigger/TrackFindingTMTT/interface/HTbase.h"
#include "L1Trigger/TrackFindingTMTT/interface/InputData.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <unordered_set>

using namespace std;

namespace tmtt {

  // Initialization.
  HTbase::HTbase(
      const Settings* settings, unsigned int iPhiSec, unsigned int iEtaReg, unsigned int nBinsX, unsigned int nBinsY)
      : settings_(settings),
        iPhiSec_(iPhiSec),
        iEtaReg_(iEtaReg),
        nBinsX_(nBinsX),
        nBinsY_(nBinsY),
        htArray_(nBinsX, nBinsY),
        optoLinkID_(this->calcOptoLinkID()) {}

  //=== Termination. Causes HT array to search for tracks etc.

  void HTbase::end() {
    // Calculate useful info about each cell in array.
    for (unsigned int i = 0; i < nBinsX_; i++) {
      for (unsigned int j = 0; j < nBinsY_; j++) {
        htArray_(i, j)->end();  // Calls HTcell::end()
      }
    }

    // Produce a list of all track candidates found in this array, each containing all the stubs on each one
    // and the track helix parameters, plus the associated truth particle (if any).
    trackCands2D_ = this->calcTrackCands2D();

    // If requested, kill those tracks in this sector that can't be read out during the time-multiplexed period, because
    // the HT has associated too many stubs to tracks.
    if (settings_->busySectorKill()) {
      trackCands2D_ = this->killTracksBusySec(trackCands2D_);
    }
  }

  //=== Number of filtered stubs in each cell summed over all cells in HT array.
  //=== If a stub appears in multiple cells, it will be counted multiple times.
  unsigned int HTbase::numStubsInc() const {
    unsigned int nStubs = 0;

    // Loop over cells in HT array.
    for (unsigned int i = 0; i < nBinsX_; i++) {
      for (unsigned int j = 0; j < nBinsY_; j++) {
        nStubs += htArray_(i, j)->numStubs();  // Calls HTcell::numStubs()
      }
    }

    return nStubs;
  }

  //=== Number of filtered stubs in HT array.
  //=== If a stub appears in multiple cells, it will be counted only once.
  unsigned int HTbase::numStubsExc() const {
    unordered_set<unsigned int> stubIDs;  // Each ID stored only once, no matter how often it is added.

    // Loop over cells in HT array.
    for (unsigned int i = 0; i < nBinsX_; i++) {
      for (unsigned int j = 0; j < nBinsY_; j++) {
        // Loop over stubs in each cells, storing their IDs.
        const vector<Stub*>& vStubs = htArray_(i, j)->stubs();  // Calls HTcell::stubs()
        for (const Stub* stub : vStubs) {
          stubIDs.insert(stub->index());
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
      nStubs += trk.stubs().size();
    }

    return nStubs;
  }

  //=== Get all reconstructed tracks that were associated to the given tracking particle.
  //=== (If the vector is empty, then the tracking particle was not reconstructed in this sector).

  vector<const L1track2D*> HTbase::assocTrackCands2D(const TP& tp) const {
    vector<const L1track2D*> assocRecoTrk;

    // Loop over track candidates, looking for those associated to given TP.
    for (const L1track2D& trk : trackCands2D_) {
      if (trk.matchedTP() != nullptr) {
        if (trk.matchedTP()->index() == tp.index())
          assocRecoTrk.push_back(&trk);
      }
    }

    return assocRecoTrk;
  }

  //=== Disable filters (used for debugging).

  void HTbase::disableBendFilter() {
    // Loop over cells in HT array.
    for (unsigned int i = 0; i < nBinsX_; i++) {
      for (unsigned int j = 0; j < nBinsY_; j++) {
        htArray_(i, j)->disableBendFilter();
      }
    }
  }

  //=== Given a range in one of the coordinates specified by coordRange, calculate the corresponding range of bins. The other arguments specify the axis. And also if some cells nominally associated to stub are to be killed.

  pair<unsigned int, unsigned int> HTbase::convertCoordRangeToBinRange(pair<float, float> coordRange,
                                                                       unsigned int nBinsAxis,
                                                                       float coordAxisMin,
                                                                       float coordAxisBinSize,
                                                                       unsigned int killSomeHTcells) const {
    float coordMin = coordRange.first;
    float coordMax = coordRange.second;
    float coordAvg = (coordRange.first + coordRange.second) / 2.;

    int iCoordBinMin, iCoordBinMax;

    //--- There are various options for doing this.
    //--- Option killSomeHTcells = 0 is the obvious one.
    //--- If killSomeHTcells > 0, then some of the cells nominally associated with the stub are killed.

    if (killSomeHTcells == 0) {
      // Take the full range of phi bins consistent with the stub.
      iCoordBinMin = floor((coordMin - coordAxisMin) / coordAxisBinSize);
      iCoordBinMax = floor((coordMax - coordAxisMin) / coordAxisBinSize);
    } else if (killSomeHTcells == 1) {
      // Use the reduced range of bins.
      // This algorithm, proposed by Ian, should reduce the rate, at the cost of some efficiency.
      const float fracCut = 0.3;
      iCoordBinMin = floor((coordMin - coordAxisMin) / coordAxisBinSize);
      iCoordBinMax = floor((coordMax - coordAxisMin) / coordAxisBinSize);
      unsigned int nbins = iCoordBinMax - iCoordBinMin + 1;
      if (nbins >= 2) {                                                      // Can't reduce range if already only 1 bin
        float lower = coordAxisMin + (iCoordBinMin + 1) * coordAxisBinSize;  // upper edge of lowest bin
        float upper = coordAxisMin + (iCoordBinMax)*coordAxisBinSize;        // lower edge of highest bin.
        // Calculate fractional amount of min and max bin that this stub uses.
        float extraLow = (lower - coordMin) / coordAxisBinSize;
        float extraUp = (coordMax - upper) / coordAxisBinSize;
        constexpr float small = 0.001;  // allow tolerance on floating point precision.
        if (min(extraLow, extraUp) < -small || max(extraLow, extraUp) > (1.0 + small))
          throw cms::Exception("LogicError") << "HTbase: convertCoordRangeToBinRange error";
        if (extraLow < fracCut && (nbins >= 3 || extraLow < extraUp))
          iCoordBinMin += 1;
        if (extraUp < fracCut && (nbins >= 3 || extraUp < extraLow))
          iCoordBinMax -= 1;
      }
    } else if (killSomeHTcells == 2) {
      // This corresponds to Thomas's firmware implementation, which can't fill more than one HT cell per column.
      iCoordBinMin = floor((coordAvg - coordAxisMin) / coordAxisBinSize);
      iCoordBinMax = iCoordBinMin;
    } else {
      throw cms::Exception("BadConfig") << "HT: invalid KillSomeHTCells option in cfg";
    }

    // Limit range to dimensions of HT array.
    iCoordBinMin = max(iCoordBinMin, 0);
    iCoordBinMax = min(iCoordBinMax, int(nBinsAxis) - 1);

    // If whole range is outside HT array, flag this by setting range to specific values with min > max.
    if (iCoordBinMin > int(nBinsAxis) - 1 || iCoordBinMax < 0) {
      iCoordBinMin = int(nBinsAxis) - 1;
      iCoordBinMax = 0;
    }

    return pair<unsigned int, unsigned int>(iCoordBinMin, iCoordBinMax);
  }

  //=== Return a list of all track candidates found in this array, giving access to all the stubs on each one
  //=== and the track helix parameters, plus the associated truth particle (if any).

  list<L1track2D> HTbase::calcTrackCands2D() const {
    list<L1track2D> trackCands2D;

    // Check if the hardware processes rows of the HT array in a specific order when outputting track candidates.
    // Currently this is by decreasing Pt for r-phi HT and unordered for r-z HT.
    const vector<unsigned int> iOrder = this->rowOrder(nBinsX_);
    bool wantOrdering = (not iOrder.empty());

    // Loop over cells in HT array.
    for (unsigned int i = 0; i < nBinsX_; i++) {
      // Access rows in specific order if required.
      unsigned int iPos = wantOrdering ? iOrder[i] : i;

      for (unsigned int j = 0; j < nBinsY_; j++) {
        if (htArray_(iPos, j)->trackCandFound()) {  // track candidate found in this cell.

          // Note if this corresponds to a merged HT cell (e.g. 2x2).
          const bool merged = htArray_(iPos, j)->mergedCell();

          // Get stubs on this track candidate.
          const vector<Stub*>& stubs = htArray_(iPos, j)->stubs();

          // And note location of cell inside HT array.
          const pair<unsigned int, unsigned int> cellLocation(iPos, j);

          // Get (q/Pt, phi0) or (tan_lambda, z0) corresponding to middle of this cell.
          const pair<float, float> helixParams2D = this->helix2Dconventional(iPos, j);

          // Create track and store it.
          trackCands2D.emplace_back(
              settings_, stubs, cellLocation, helixParams2D, iPhiSec_, iEtaReg_, optoLinkID_, merged);
        }
      }
    }

    return trackCands2D;
  }

}  // namespace tmtt
