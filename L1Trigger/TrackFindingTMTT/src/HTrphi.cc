#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"
#include "L1Trigger/TrackFindingTMTT/interface/InputData.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/PrintL1trk.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
#include <limits>
#include <atomic>
#include <sstream>
#include <mutex>

using namespace std;

namespace tmtt {

  //=== The r-phi Hough Transform array for a single (eta,phi) sector.
  //===
  //=== Its axes are (q/Pt, phiTrk), where phiTrk is the phi at which the track crosses a
  //=== user-configurable radius from the beam-line.

  //=== Initialise

  HTrphi::HTrphi(const Settings* settings,
                 unsigned int iPhiSec,
                 unsigned int iEtaReg,
                 float etaMinSector,
                 float etaMaxSector,
                 float phiCentreSector,
                 HTrphi::ErrorMonitor* errMon)
      : HTbase(settings, iPhiSec, iEtaReg, settings->houghNbinsPt(), settings->houghNbinsPhi()),
        invPtToDphi_((settings->invPtToDphi())),
        shape_(static_cast<HTshape>(settings->shape())),  // shape of HT cells

        //--- Specification of HT q/Pt axis.
        maxAbsQoverPtAxis_(1. / settings->houghMinPt()),  // Max. |q/Pt| covered by  HT array.
        nBinsQoverPtAxis_(settings->houghNbinsPt()),      // No. of bins in HT array in q/Pt.
        binSizeQoverPtAxis_(2 * maxAbsQoverPtAxis_ / nBinsQoverPtAxis_),

        //--- Specification of HT phiTrk axis
        // (phiTrk corresponds to phi where track crosses radius = chosenRofPhi_).
        chosenRofPhi_(settings->chosenRofPhi()),
        phiCentreSector_(phiCentreSector),                           // Centre of phiTrk sector.
        maxAbsPhiTrkAxis_(M_PI / float(settings->numPhiSectors())),  // Half-width of phiTrk axis in HT array.
        nBinsPhiTrkAxis_(settings->houghNbinsPhi()),                 // No. of bins in HT array phiTrk
        binSizePhiTrkAxis_(2 * maxAbsPhiTrkAxis_ / nBinsPhiTrkAxis_),
        errMon_(errMon) {
    // Deal with unusually shaped HT cells.
    if (shape_ != HTshape::square)
      binSizeQoverPtAxis_ = 2. * maxAbsQoverPtAxis_ / (nBinsQoverPtAxis_ - 1.);
    if (shape_ == HTshape::hexagon)
      binSizePhiTrkAxis_ = 2. * maxAbsPhiTrkAxis_ / (nBinsPhiTrkAxis_ - 1. / 6.);
    else if (shape_ == HTshape::diamond)
      binSizePhiTrkAxis_ = 2. * maxAbsPhiTrkAxis_ / (nBinsPhiTrkAxis_ - 1. / 2.);

    // Optionally merge 2x2 neighbouring cells into a single cell at low Pt, to reduce efficiency loss due to
    // scattering. (Do this if either of options EnableMerge2x2 or MiniHTstage are enabled.
    // N.B These two options are never both enabled).
    enableMerge2x2_ = (settings->enableMerge2x2() || settings->miniHTstage());
    if (settings->miniHTstage()) {
      // Mini-HT stage cfg: Merge all bins, irrespective of Pt.
      minInvPtToMerge2x2_ = 0.;
    } else {
      // Merged cells cfg: Merge bins below specified Pt threshold.
      minInvPtToMerge2x2_ = 1. / (settings->maxPtToMerge2x2());
      if (minInvPtToMerge2x2_ > maxAbsQoverPtAxis_)
        enableMerge2x2_ = false;
    }

    // Merging 2x2 cells into 1 merged cell is only allowed if HT array dimensions are even.
    // (This restriction could be removed along q/Pt axis, since there are also unmerged cells there. But this
    // would require correcting the code after each called to mergedCell() below, since
    //  "if (i%2 == 1) iStore = i - 1" not correct in this case).
    if (enableMerge2x2_ && (nBinsQoverPtAxis_ % 2 != 0 || nBinsPhiTrkAxis_ % 2 != 0))
      throw cms::Exception("BadConfig") << "HTrphi: You are not allowed to set EnableMerge2x2 or MiniHTstage = True if "
                                           "you have an odd number of bins "
                                           "in r-phi HT array "
                                        << nBinsQoverPtAxis_ << " " << nBinsPhiTrkAxis_;

    //--- Other options used when filling the HT.

    // Don't fill all HT cells nominally crossed by line corresponding to stub.
    killSomeHTCellsRphi_ = settings->killSomeHTCellsRphi();

    // Used to kill excess stubs or tracks that can't be transmitted within time-multiplexed period.
    nReceivedStubs_ = 0;
    busyInputSectorKill_ = settings_->busyInputSectorKill();  // Kill excess stubs going fron GP to HT?
    busySectorKill_ = settings_->busySectorKill();            // Kill excess tracks flowing out of HT?
    // Max. num. of stubs that can be sent from GP to HT within TM period
    busyInputSectorNumStubs_ = settings_->busyInputSectorNumStubs();
    // Max. num. of stubs that can be sent out of HT within TM period
    busySectorNumStubs_ = settings_->busySectorNumStubs();
    // or individual m bin (=q/Pt) ranges to be output to optical links.
    busySectorMbinRanges_ = settings_->busySectorMbinRanges();
    // Specifies which m bins should be grouped together by BusySectorMbinRanges. If empty, then they are grouped in order 0,1,2,3,4,5 ...
    busySectorMbinOrder_ = settings_->busySectorMbinOrder();
    // m bin ranges option disabled if vector empty
    busySectorUseMbinRanges_ = (not busySectorMbinRanges_.empty());
    busySectorUseMbinOrder_ = (not busySectorMbinOrder_.empty());

    bool rescaleMbins = false;
    if (busySectorUseMbinRanges_) {
      // Check if the total number of bins specified in cfg option BusySectorMbinRanges corresponds
      // to the number of m bins (q/Pt) in the HT. If not, determine how much the ranges must be scaled
      // to make this true.
      unsigned int nTotalBins = 0;
      for (unsigned int j = 0; j < busySectorMbinRanges_.size(); j++) {
        nTotalBins += busySectorMbinRanges_[j];
      }
      rescaleMbins = (nTotalBins != nBinsQoverPtAxis_);
      // No rescaling allowed with MBinOrder option.
      if (rescaleMbins && busySectorUseMbinOrder_)
        throw cms::Exception("BadConfig") << "HTrphi: BusySectorUserMbin error";
      float rescaleFactor = rescaleMbins ? float(nBinsQoverPtAxis_) / float(nTotalBins) : 1.;
      // Find lower and upper inclusive limits of each m bin range to be sent to a separate optical link.
      busySectorMbinLow_.resize(busySectorMbinRanges_.size());
      busySectorMbinHigh_.resize(busySectorMbinRanges_.size());
      float mBinSum = 0.;
      for (unsigned int i = 0; i < busySectorMbinRanges_.size(); i++) {
        busySectorMbinLow_[i] = std::round(mBinSum);
        busySectorMbinHigh_[i] = std::round(mBinSum + rescaleFactor * busySectorMbinRanges_[i]) - 1;
        mBinSum += rescaleFactor * busySectorMbinRanges_[i];
      }
    }
    //
    for (unsigned int i = 0; i < nBinsQoverPtAxis_; i++) {
      for (unsigned int j = 0; j < nBinsPhiTrkAxis_; j++) {
        pair<float, float> helix = this->helix2Dconventional(i, j);  // Get track params at centre of cell.
        float qOverPt = helix.first;
        // Check if this cell is merged with its neighbours (as in low Pt region).
        bool mergedCell = false;
        if (enableMerge2x2_ && this->mergedCell(i, j))
          mergedCell = true;
        // Initialize each cell in HT array.
        HTbase::htArray_(i, j) =
            std::make_unique<HTcell>(settings, iPhiSec, iEtaReg, etaMinSector, etaMaxSector, qOverPt, i, mergedCell);
      }
    }

    std::stringstream text;
    text << "\n";
    text << "=== R-PHI HOUGH TRANSFORM AXES RANGES: abs(q/Pt) < " << maxAbsQoverPtAxis_ << " & abs(track-phi) < "
         << maxAbsPhiTrkAxis_ << " ===\n";
    text << "=== R-PHI HOUGH TRANSFORM ARRAY SIZE: q/Pt bins = " << nBinsQoverPtAxis_
         << " & track-phi bins = " << nBinsPhiTrkAxis_ << " ===\n";
    text << "=== R-PHI HOUGH TRANSFORM BIN SIZE: BIN(q/Pt) = " << binSizeQoverPtAxis_
         << " & BIN(track-phi) = " << binSizePhiTrkAxis_ << " ===\n\n";
    if (busySectorKill_ && busySectorUseMbinRanges_ && rescaleMbins) {
      text << "=== R-PHI HOUGH TRANSFORM WARNING: Rescaled m bin ranges specified by cfg parameter "
              "BusySectorMbinRanges, as they were inconsistent with total number of m bins in HT.\n";
      text << "=== Rescaled values for BusySectorMbinRanges =";
      for (unsigned int i = 0; i < busySectorMbinRanges_.size(); i++) {
        text << " " << (busySectorMbinHigh_[i] - busySectorMbinLow_[i] + 1);
      }
    }
    text << "\n";
    static std::once_flag printOnce;
    std::call_once(
        printOnce, [](string t) { PrintL1trk() << t; }, text.str());

    // Note helix parameters at the centre of each HT cell.
    cellCenters_.clear();
    for (unsigned int m = 0; m < nBinsQoverPtAxis_; m++) {
      std::vector<std::pair<float, float> > binCenters;
      for (unsigned int c = 0; c < nBinsPhiTrkAxis_; c++)
        binCenters.push_back(this->helix2Dhough(m, c));
      cellCenters_.push_back(binCenters);
    }
  }

  //=== Add stub to HT array.
  //=== If eta subsectors are being used within each sector, specify which ones the stub is compatible with.

  void HTrphi::store(Stub* stub, const vector<bool>& inEtaSubSecs) {
    // Optionally, only store stubs that can be sent from GP to HT within TM period.
    if ((!busyInputSectorKill_) || (nReceivedStubs_ < busyInputSectorNumStubs_)) {
      nReceivedStubs_++;

      unsigned int jPhiTrkBinMinLast = 0;  // Used for error checking
      unsigned int jPhiTrkBinMaxLast = 99999;

      // Loop over q/Pt related bins in HT array.
      for (unsigned int i = 0; i < nBinsQoverPtAxis_; i++) {
        if (shape_ == HTshape::square) {
          //--- This is a traditional HT with square cells.

          // In this q/Pt bin, find the range of phi bins that this stub is consistent with.
          pair<unsigned int, unsigned int> jRange = this->iPhiRange(stub, i);
          unsigned int jPhiTrkBinMin = jRange.first;
          unsigned int jPhiTrkBinMax = jRange.second;

          // Store stubs in these cells.
          for (unsigned int j = jPhiTrkBinMin; j <= jPhiTrkBinMax; j++) {
            bool canStoreStub = true;
            unsigned int iStore = i;
            unsigned int jStore = j;

            // Optionally merge 2x2 neighbouring cells into a single cell at low Pt, to reduce efficiency loss
            // due to scattering.
            if (enableMerge2x2_) {
              // Check if this cell is merged with its neighbours (as in low Pt region).
              if (this->mergedCell(i, j)) {
                // Get location of cell that this cell is merged into (iStore, jStore).
                // Calculation assumes HT array has even number of bins in both dimensions.
                if (i % 2 == 1)
                  iStore = i - 1;
                if (j % 2 == 1)
                  jStore = j - 1;
                // If this stub was already stored in this merged 2x2 cell, then don't store it again.
                if (HTbase::htArray_(iStore, jStore)->stubStoredInCell(stub))
                  canStoreStub = false;
              }
            }

            if (canStoreStub)
              HTbase::htArray_(iStore, jStore)->store(stub, inEtaSubSecs);  // Calls HTcell::store()
          }

          // Check that limitations of firmware would not prevent stub being stored correctly in this HT column.
          if (errMon_ != nullptr) {
            this->countFirmwareErrors(i, jPhiTrkBinMin, jPhiTrkBinMax, jPhiTrkBinMinLast, jPhiTrkBinMaxLast);
            jPhiTrkBinMinLast = jPhiTrkBinMin;
            jPhiTrkBinMaxLast = jPhiTrkBinMax;
          }

        } else {
          //--- This is are novel HT with unusual shaped cells.

          if (shape_ == HTshape::diamond) {
            //--- This HT has diamond shaped cells.

            float qOverPtBin = -maxAbsQoverPtAxis_ + i * binSizeQoverPtAxis_;
            float phiTrk = reco::deltaPhi(stub->phi(), phiCentreSector_) +
                           invPtToDphi_ * qOverPtBin * (stub->r() - chosenRofPhi_) + maxAbsPhiTrkAxis_;
            if (i % 2 == 0)
              phiTrk += binSizePhiTrkAxis_ / 2.;
            unsigned int binCenter = std::floor(phiTrk / binSizePhiTrkAxis_);
            if (binCenter < nBinsPhiTrkAxis_)
              HTbase::htArray_(i, binCenter)->store(stub, inEtaSubSecs);

          } else if (shape_ == HTshape::hexagon) {
            //--- This HT has hexagonal cells (with two of its sides parallel to the phi axis).

            float qOverPtBin = -maxAbsQoverPtAxis_ + i * binSizeQoverPtAxis_;
            float qOverPtBinVar = binSizeQoverPtAxis_;
            float phiTrk = reco::deltaPhi(stub->phi(), phiCentreSector_) +
                           invPtToDphi_ * qOverPtBin * (stub->r() - chosenRofPhi_) + maxAbsPhiTrkAxis_;
            float phiTrkVar = invPtToDphi_ * qOverPtBinVar * std::abs(stub->r() - chosenRofPhi_);
            float phiTrkMin = phiTrk - phiTrkVar;
            float phiTrkMax = phiTrk + phiTrkVar;
            if (i % 2 == 0)
              phiTrk += binSizePhiTrkAxis_ / 6.;
            else {
              phiTrk -= binSizePhiTrkAxis_ / 3.;
              phiTrkMin -= binSizePhiTrkAxis_ / 2.;
              phiTrkMax -= binSizePhiTrkAxis_ / 2.;
            }
            unsigned int iCenter = std::floor(phiTrk / binSizePhiTrkAxis_ * 3.);
            unsigned int iMin = std::floor(phiTrkMin / binSizePhiTrkAxis_ * 3.);
            unsigned int iMax = std::floor(phiTrkMax / binSizePhiTrkAxis_ * 3.);
            std::pair<bool, unsigned int> binCenter;
            std::pair<bool, unsigned int> binMin;
            std::pair<bool, unsigned int> binMax;
            binCenter.second = iCenter / 3;
            binMin.second = iMin / 3;
            binMax.second = iMax / 3;
            binCenter.first = !(iCenter % 3 == 2);
            binMin.first = (iMin % 3 == 0);
            binMax.first = (iMax % 3 == 0);
            if (binCenter.first && binCenter.second < nBinsPhiTrkAxis_)
              HTbase::htArray_(i, binCenter.second)->store(stub, inEtaSubSecs);
            else if (binMin.first && binMin.second < nBinsPhiTrkAxis_)
              HTbase::htArray_(i, binMin.second)->store(stub, inEtaSubSecs);
            else if (binMax.first && binMax.second < nBinsPhiTrkAxis_)
              HTbase::htArray_(i, binMax.second)->store(stub, inEtaSubSecs);

          } else if (shape_ == HTshape::brick) {
            //--- This HT has square cells with alternate rows shifted horizontally by 0.5*cell_width.

            float qOverPtBin = -maxAbsQoverPtAxis_ + i * binSizeQoverPtAxis_;
            float qOverPtBinVar = binSizeQoverPtAxis_;
            float phiTrk = reco::deltaPhi(stub->phi(), phiCentreSector_) +
                           invPtToDphi_ * qOverPtBin * (stub->r() - chosenRofPhi_) + maxAbsPhiTrkAxis_;
            float phiTrkVar = invPtToDphi_ * qOverPtBinVar * std::abs(stub->r() - chosenRofPhi_);
            float phiTrkMin = phiTrk - phiTrkVar;
            float phiTrkMax = phiTrk + phiTrkVar;
            unsigned int iMin = std::floor(phiTrkMin / binSizePhiTrkAxis_ * 2.);
            unsigned int iMax = std::floor(phiTrkMax / binSizePhiTrkAxis_ * 2.);
            std::pair<bool, unsigned int> binMin;
            std::pair<bool, unsigned int> binMax;
            binMin.second = iMin / 2;
            binMax.second = iMax / 2;
            binMin.first = (iMin % 2 == i % 2);
            binMax.first = (iMax % 2 == i % 2);
            if (binMin.first && binMin.second < nBinsPhiTrkAxis_)
              HTbase::htArray_(i, binMin.second)->store(stub, inEtaSubSecs);
            else if (binMax.first && binMax.second < nBinsPhiTrkAxis_)
              HTbase::htArray_(i, binMax.second)->store(stub, inEtaSubSecs);
          }
        }
      }
      // Note max. |gradient| that the line corresponding to any stub in any of the r-phi HT arrays could have.
      // Firmware assumes this should not exceed 1.0;
      if (errMon_ != nullptr) {
        errMon_->maxLineGradient = max(errMon_->maxLineGradient.load(), this->calcLineGradArray(stub->r()));
      }
    }
  }

  //=== Determine the m-bin (q/pt) range the specified track is in. (Used if outputting each m bin range on a different opto-link).

  unsigned int HTrphi::getMbinRange(const L1track2D& trk) const {
    if (busySectorUseMbinRanges_) {
      unsigned int mBin = trk.cellLocationHT().first;
      unsigned int mBinOrder;
      if (busySectorUseMbinOrder_) {
        // User wants to group bins in a wierd order.
        mBinOrder = 99999;
        for (unsigned int k = 0; k < busySectorMbinOrder_.size(); k++) {
          if (mBin == busySectorMbinOrder_[k])
            mBinOrder = k;
        }
        if (mBinOrder == 99999)
          throw cms::Exception("LogicError") << "HTrphi::getMbinRange() mBinOrder calculation wrong.";
      } else {
        // User grouping bins in numerical order 0,1,2,3,4,5...
        mBinOrder = mBin;
      }
      for (unsigned int i = 0; i < busySectorMbinRanges_.size(); i++) {
        if (mBinOrder >= busySectorMbinLow_[i] && mBinOrder <= busySectorMbinHigh_[i])
          return i;
      }
      throw cms::Exception("LogicError") << "HTrphi::getMbinRange() messed up";
    } else {
      return 0;
    }
  }

  //=== For a given Q/Pt bin, find the range of phi bins that a given stub is consistent with.
  //=== Return as a pair (min bin, max bin)
  //=== If it range lies outside the HT array, then the min bin will be set larger than the max bin.

  pair<unsigned int, unsigned int> HTrphi::iPhiRange(const Stub* stub, unsigned int iQoverPtBin, bool debug) const {
    // Note q/Pt value corresponding to centre of this bin.
    float qOverPtBin = -maxAbsQoverPtAxis_ + (iQoverPtBin + 0.5) * binSizeQoverPtAxis_;
    // Note change in this q/Pt value needed to reach either edge of the bin.
    float qOverPtBinVar = 0.5 * binSizeQoverPtAxis_;

    // Reducing effective bin width can reduce fake rate.
    //qOverPtVar = 0.4*binSizeQoverPtAxis_;

    // Calculate range of track-phi that would allow a track in this q/Pt range to pass through the stub.
    float phiTrk = stub->phi() + invPtToDphi_ * qOverPtBin * (stub->r() - chosenRofPhi_);
    // The next line does the phiTrk calculation without the usual approximation, but it doesn't
    // improve performance.
    //float phiTrk    = stub->phi() + asin(invPtToDphi_ * qOverPtBin * stub->r()) - asin(invPtToDphi_ * qOverPtBin * chosenRofPhi_);
    float phiTrkVar = invPtToDphi_ * qOverPtBinVar * std::abs(stub->r() - chosenRofPhi_);
    float phiTrkMin = phiTrk - phiTrkVar;
    float phiTrkMax = phiTrk + phiTrkVar;

    float deltaPhiMin = reco::deltaPhi(phiTrkMin, phiCentreSector_);  // Offset to centre of sector.
    float deltaPhiMax = reco::deltaPhi(phiTrkMax, phiCentreSector_);
    pair<float, float> phiTrkRange(deltaPhiMin, deltaPhiMax);

    // Determine which HT array cell range in track-phi this range "phiTrkRange" corresponds to.
    pair<unsigned int, unsigned int> iPhiTrkBinRange = this->HTbase::convertCoordRangeToBinRange(
        phiTrkRange, nBinsPhiTrkAxis_, (-maxAbsPhiTrkAxis_), binSizePhiTrkAxis_, killSomeHTCellsRphi_);

    return iPhiTrkBinRange;
  }

  //=== Check that limitations of firmware would not prevent stub being stored correctly in this HT column.

  void HTrphi::countFirmwareErrors(unsigned int iQoverPtBin,
                                   unsigned int jPhiTrkBinMin,
                                   unsigned int jPhiTrkBinMax,
                                   unsigned int jPhiTrkBinMinLast,
                                   unsigned int jPhiTrkBinMaxLast) {
    // Only do check if stub is being stored somewhere in this HT column.
    if (jPhiTrkBinMax >= jPhiTrkBinMin) {
      //--- Remaining code below checks that firmware could successfully store this stub in this column.
      //   (a) Does cell lie NE, E or SE of cell filled in previous column?
      bool OK_a = (jPhiTrkBinMin + 1 >= jPhiTrkBinMinLast) && (jPhiTrkBinMax <= jPhiTrkBinMaxLast + 1);
      //   (b) Are no more than 2 cells filled in this column
      bool OK_b = (jPhiTrkBinMax - jPhiTrkBinMin + 1 <= 2);

      if (!OK_a)
        errMon_->numErrorsTypeA++;
      if (!OK_b)
        errMon_->numErrorsTypeB++;
      errMon_->numErrorsNorm++;  // No. of times a stub is added to an HT column.
    }
  }

  //=== Get the values of the track helix params corresponding to middle of a specified HT cell (i,j).
  //=== The helix parameters returned will be those corresponding to the two axes of the HT array.
  //=== So they might be (q/pt, phi0) or (q/pt, phi65) etc. depending on the configuration.

  pair<float, float> HTrphi::helix2Dhough(unsigned int i, unsigned int j) const {
    unsigned int qOverPtBin = i;
    unsigned int phiTrkBin = j;

    // If using merged 2x2 cells in low Pt parts of array, must correct for this.
    bool merged = false;
    if (enableMerge2x2_) {
      // Check if this cell is merged with its neighbours (as in low Pt region).
      if (this->mergedCell(i, j)) {
        merged = true;
        // Get location of cell that this cell is merged into (iStore, jStore).
        // Calculation assumes HT array has even number of bins in both dimensions.
        if (i % 2 == 1)
          qOverPtBin = i - 1;
        if (j % 2 == 1)
          phiTrkBin = j - 1;
      }
    }

    float qOverPtBinCenter = .5;
    float phiTrkBinCenter = .5;

    if (shape_ != HTshape::square) {
      qOverPtBinCenter = 0.;

      float evenPhiPos = 0., oddPhiPos = 0.;
      if (shape_ == HTshape::hexagon) {
        evenPhiPos = 1. / 6.;
        oddPhiPos = 2. / 3.;
      } else if (shape_ == HTshape::diamond) {
        evenPhiPos = 0.;
        oddPhiPos = 0.5;
      } else if (shape_ == HTshape::brick) {
        evenPhiPos = 0.25;
        oddPhiPos = 0.75;
      }
      phiTrkBinCenter = (qOverPtBin % 2 == 0) ? evenPhiPos : oddPhiPos;
    }

    float qOverPt = -maxAbsQoverPtAxis_ + (qOverPtBin + qOverPtBinCenter) * binSizeQoverPtAxis_;
    float phiTrk = -maxAbsPhiTrkAxis_ + (phiTrkBin + phiTrkBinCenter) * binSizePhiTrkAxis_;

    if (merged) {
      qOverPt += 0.5 * binSizeQoverPtAxis_;
      phiTrk += 0.5 * binSizePhiTrkAxis_;
    }

    // Correct phiTrk to centre of sector, taking care of 2*pi wrapping
    phiTrk = reco::deltaPhi(phiTrk + phiCentreSector_, 0.);
    return pair<float, float>(qOverPt, phiTrk);
  }

  //=== Get the values of the track helix params corresponding to middle of a specified HT cell (i,j).
  //=== The helix parameters returned will be always be (q/pt, phi0), irrespective of how the axes
  //=== of the HT array are defined.

  pair<float, float> HTrphi::helix2Dconventional(unsigned int i, unsigned int j) const {
    // Get the helix parameters corresponding to the axes definitions of the HT.
    pair<float, float> helix2Dht = this->helix2Dhough(i, j);
    // Convert to the conventionally agreed pair of helix parameters, (q/pt, phi0).
    float qOverPt = helix2Dht.first;  // easy
    // If HT defined track phi other than at r=0, must correct to get phi0. Allow for 2*pi wrapping of phi.
    float phi0 = reco::deltaPhi(helix2Dht.second + invPtToDphi_ * chosenRofPhi_ * qOverPt, 0.);
    return pair<float, float>(qOverPt, phi0);
  }

  //=== Which cell in HT array should this TP be in, based on its true trajectory?
  //=== (If TP is outside HT array, it it put in the closest bin inside it).

  pair<unsigned int, unsigned int> HTrphi::trueCell(const TP* tp) const {
    // Get HT axis variables corresponding to this TP.
    float qOverPt = tp->qOverPt();
    float phiTrk = tp->trkPhiAtR(chosenRofPhi_);
    // Measure phi relative to centre of sector.
    float deltaPhi = reco::deltaPhi(phiTrk, phiCentreSector_);
    // Convert to bin numbers inside HT array.
    int iQoverPt = floor((qOverPt - (-maxAbsQoverPtAxis_)) / binSizeQoverPtAxis_);
    int iPhiTrk = floor((deltaPhi - (-maxAbsPhiTrkAxis_)) / binSizePhiTrkAxis_);
    // Check if this cell was within the HT array.
    if (iQoverPt >= 0 && iQoverPt < int(nBinsQoverPtAxis_) && iPhiTrk >= 0 && iPhiTrk < int(nBinsPhiTrkAxis_)) {
      // Check if this cell is merged with its neighbours (as in low Pt region), and if so return merged cell location.
      // New: because 2nd stage mini HT may recreate tracks from merged cells with finer cell granularity, one can't predict
      //      if a merged cell was used to create a track merely by looking at its cell location.
      //      So instead ask L1track3D, which knows if it was created from a merged HT cell or not.
      ;
    } else {
      // TP is not in this HT array at all. Flag this by setting "outside" bin index to 0 (Nbins) if outside array below (above).
      if (iQoverPt < 0)
        iQoverPt = 0;
      if (iQoverPt >= int(nBinsQoverPtAxis_))
        iQoverPt = nBinsQoverPtAxis_ - 1;
      if (iPhiTrk < 0)
        iPhiTrk = 0;
      if (iPhiTrk >= int(nBinsPhiTrkAxis_))
        iPhiTrk = nBinsPhiTrkAxis_ - 1;
    }
    return pair<unsigned int, unsigned int>(iQoverPt, iPhiTrk);
  }

  //=== Which cell in HT array should this fitted track be in, based on its fitted trajectory?
  //=== Always uses beam-spot constrained trajectory if available.
  //=== (If fitted track is outside HT array, it it put in the closest bin inside it).

  pair<unsigned int, unsigned int> HTrphi::cell(const L1fittedTrack* fitTrk) const {
    bool beamConstraint = fitTrk->done_bcon();  // Is beam-spot constraint available? (e.g. 5 param helix fit)
    // Get HT axis variables corresponding to this fitted track.
    float qOverPt = beamConstraint ? fitTrk->qOverPt_bcon() : fitTrk->qOverPt();
    // Convert phi0 to phi at chosen radius used in HT.
    float phiTrk = fitTrk->phiAtChosenR(beamConstraint);
    // Measure phi relative to centre of sector.
    float deltaPhi = reco::deltaPhi(phiTrk, phiCentreSector_);
    // Convert to bin numbers inside HT array.
    int iQoverPt = 999999;
    int iPhiTrk = 999999;

    if (shape_ == HTshape::square) {
      //--- This is a traditional HT with square cells.

      iQoverPt = floor((qOverPt - (-maxAbsQoverPtAxis_)) / binSizeQoverPtAxis_);
      iPhiTrk = floor((deltaPhi - (-maxAbsPhiTrkAxis_)) / binSizePhiTrkAxis_);

      // Check if this cell was within the HT array.
      if (iQoverPt >= 0 && iQoverPt < int(nBinsQoverPtAxis_) && iPhiTrk >= 0 && iPhiTrk < int(nBinsPhiTrkAxis_)) {
        // Check if this cell is merged with its neighbours (as in low Pt region), and if so return merged cell location.
        // New: because 2nd stage mini HT may recreate tracks from merged cells with finer cell granularity, one can't predict
        //      if a merged cell was used to create a track merely by looking at its cell location.
        //      So instead ask L1track3D, which knows if it was created from a merged HT cell or not.
        ;
      } else {
        // Fitted track is not in this HT array at all. Flag this by setting "outside" bin index to 0 (Nbins-1) if outside array below (above).
        if (iQoverPt < 0)
          iQoverPt = 0;
        if (iQoverPt >= int(nBinsQoverPtAxis_))
          iQoverPt = nBinsQoverPtAxis_ - 1;
        if (iPhiTrk < 0)
          iPhiTrk = 0;
        if (iPhiTrk >= int(nBinsPhiTrkAxis_))
          iPhiTrk = nBinsPhiTrkAxis_ - 1;
      }

    } else {
      //--- This is are novel HT with unusual shaped cells.

      float minD = std::numeric_limits<float>::infinity();
      float d(0);
      unsigned int m(0);
      for (const auto& binCenters : cellCenters_) {
        unsigned int c(0);
        for (auto cellCenter : binCenters) {
          d = std::pow((cellCenter.first - qOverPt) / (float)binSizeQoverPtAxis_, 2) +
              std::pow((cellCenter.second - phiTrk) / (float)binSizePhiTrkAxis_, 2);
          if (d < minD) {
            minD = d;
            iQoverPt = m;
            iPhiTrk = c;
          }
          c++;
        }
        m++;
      }
      // Fitted track is not in this HT array at all. Flag this by setting "outside" bin index to 0 (Nbins-1) if outside array below (above).
      if (iQoverPt < 0)
        iQoverPt = 0;
      if (iQoverPt >= int(nBinsQoverPtAxis_))
        iQoverPt = nBinsQoverPtAxis_ - 1;
      if (iPhiTrk < 0)
        iPhiTrk = 0;
      if (iPhiTrk >= int(nBinsPhiTrkAxis_))
        iPhiTrk = nBinsPhiTrkAxis_ - 1;
    }

    return pair<unsigned int, unsigned int>(iQoverPt, iPhiTrk);
  }

  //=== Check if specified cell is merged with its 2x2 neighbours into a single cell,
  //=== as it is in low Pt region.

  bool HTrphi::mergedCell(unsigned int iQoverPtBin, unsigned int jPhiTrkBin) const {
    bool merge = false;

    if (enableMerge2x2_) {
      unsigned int i = iQoverPtBin;
      //unsigned int j = jPhiTrkBin;

      // Calculate number of merged bins on each q/Pt side of array.
      float fMergeBins = (maxAbsQoverPtAxis_ - minInvPtToMerge2x2_) / (2. * binSizeQoverPtAxis_);
      // Number of unmerged bins this corresponds to, which must be even, since each merged bin comprises two normal q/pt bins.
      unsigned int numQoverPtBinsToMerge = 2 * min((unsigned int)(std::round(fMergeBins)), (nBinsQoverPtAxis_ / 4));
      const float small = 0.001;
      if (minInvPtToMerge2x2_ < small && (unsigned int)(std::round(2. * fMergeBins)) % 2 == 1)
        numQoverPtBinsToMerge++;
      unsigned int iB = (nBinsQoverPtAxis_ - 1) - i;  // Count backwards across array.
      if (min(i, iB) < numQoverPtBinsToMerge)
        merge = true;
    }

    return merge;
  }

  //=== Calculate line |gradient| of stubs in HT array, so can check it doesn't exceed 1.

  float HTrphi::calcLineGradArray(float r) const {
    float grad = std::abs(invPtToDphi_ * (r - chosenRofPhi_));
    // Convert it to units of bin width.
    grad *= binSizeQoverPtAxis_ / binSizePhiTrkAxis_;
    if (shape_ == HTshape::hexagon)
      grad *= 3.;
    else if (shape_ == HTshape::diamond)
      grad *= 2.;
    else if (shape_ == HTshape::brick)
      grad *= 4.;
    return grad;
  }

  //=== If requested, kill those tracks in this sector that can't be read out during the time-multiplexed period, because
  //=== the HT has associated too many stubs to tracks.

  list<L1track2D> HTrphi::killTracksBusySec(const list<L1track2D>& tracks) const {
    list<L1track2D> outTracks;

    if (busySectorKill_) {
      unsigned int nStubsOut = 0;  // #stubs assigned to tracks in this sector.
      // #stubs assigned to each m bin range in this sector.
      vector<unsigned int> nStubsOutInRange(busySectorMbinRanges_.size(), 0);

      for (const L1track2D& trk : tracks) {
        bool keep = true;
        unsigned int nStubs = trk.numStubs();  // #stubs on this track.
        if (busySectorUseMbinRanges_) {  // Are tracks from different m bin ranges output seperately to increase bandwidth?
          unsigned int mBinRange = this->getMbinRange(trk);  // Which m bin range is this track in?
          nStubsOutInRange[mBinRange] += nStubs;
          if (nStubsOutInRange[mBinRange] > busySectorNumStubs_)
            keep = false;
        } else {
          nStubsOut += nStubs;
          if (nStubsOut > busySectorNumStubs_)
            keep = false;
        }

        if (keep)
          outTracks.push_back(trk);
      }

    } else {
      outTracks = tracks;
    }

    return outTracks;
  }

  //=== Define the order in which the hardware processes rows of the HT array when it outputs track candidates.
  //=== Currently corresponds to highest Pt tracks first.
  //=== If two tracks have the same Pt, the -ve charge one is output before the +ve charge one.

  vector<unsigned int> HTrphi::rowOrder(unsigned int numRows) const {
    vector<unsigned int> iOrder;

    // Logic slightly different depending on whether HT array has even or odd number of rows.
    const bool oddNumRows = (numRows % 2 == 1);

    // This selects middle rows first before moving to exterior ones.
    if (oddNumRows) {
      unsigned int middleRow = (numRows - 1) / 2;
      iOrder.push_back(middleRow);
      for (unsigned int i = 1; i <= (numRows - 1) / 2; i++) {
        iOrder.push_back(middleRow - i);  // -ve charge
        iOrder.push_back(middleRow + i);  // +ve charge
      }
    } else {
      unsigned int startRowPos = numRows / 2;
      unsigned int startRowNeg = startRowPos - 1;
      for (unsigned int i = 0; i < numRows / 2; i++) {
        iOrder.push_back(startRowNeg - i);  // -ve charge
        iOrder.push_back(startRowPos + i);  // +ve charge
      }
    }

    return iOrder;
  }

}  // namespace tmtt
