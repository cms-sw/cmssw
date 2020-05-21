#ifndef L1Trigger_TrackFindingTMTT_HTrphi_h
#define L1Trigger_TrackFindingTMTT_HTrphi_h

#include "L1Trigger/TrackFindingTMTT/interface/HTbase.h"

#include <vector>
#include <list>
#include <utility>
#include <atomic>

//=== The r-phi Hough Transform array for a single (eta,phi) sector.
//===
//=== Its axes are (q/Pt, phiTrk), where phiTrk is the phi at which the track crosses a
//=== user-configurable radius from the beam-line.

namespace tmtt {

  class Settings;
  class Stub;
  class TP;
  class L1fittedTrack;

  class HTrphi : public HTbase {
  public:
    enum class HTshape { square, diamond, hexagon, brick };

    //--- Error monitoring

    struct ErrorMonitor {
      // Maximum |gradient| of line corresponding to any stub. FW assumes it's < 1.0.
      std::atomic<float> maxLineGradient;
      // Error count when stub added to cell not NE, E or SE of cell stub added to in previous HT column.
      std::atomic<unsigned int> numErrorsTypeA;
      // Error count when stub added to more than 2 cells in one HT column
      std::atomic<unsigned int> numErrorsTypeB;
      // Error count normalisation
      std::atomic<unsigned int> numErrorsNorm;
    };

    // Initialization with sector number, eta range covered by sector and phi coordinate of its centre.
    HTrphi(const Settings* settings,
           unsigned int iPhiSec,
           unsigned int iEtaReg,
           float etaMinSector,
           float etaMaxSector,
           float phiCentreSector,
           ErrorMonitor* errMon = nullptr);

    ~HTrphi() override = default;

    // Add stub to HT array.
    // If eta subsectors are being used within each sector, specify which ones the stub is compatible with.
    void store(Stub* stub, const std::vector<bool>& inEtaSubSecs);

    // Termination. Causes HT array to search for tracks etc.
    // ... function end() is in base class ...

    //=== Info about track candidates found.

    // ... is available via base class ...

    //=== Utilities

    // Get the values of the track helix params corresponding to middle of a specified HT cell (i,j).
    // The helix parameters returned will be those corresponding to the two axes of the HT array.
    // So they might be (q/pt, phi0) or (q/pt, phi65) etc. depending on the configuration.
    std::pair<float, float> helix2Dhough(unsigned int i, unsigned int j) const override;

    // Get the values of the track helix params corresponding to middle of a specified HT cell (i,j).
    // The helix parameters returned will be always be (q/pt, phi0), irrespective of how the axes
    // of the HT array are defined.
    std::pair<float, float> helix2Dconventional(unsigned int i, unsigned int j) const override;

    // Which cell in HT array should this TP be in, based on its true trajectory?
    // (If TP is outside HT array, it it put in the closest bin inside it).
    std::pair<unsigned int, unsigned int> trueCell(const TP* tp) const override;

    // Which cell in HT array should this fitted track be in, based on its fitted trajectory?
    // Always uses beam-spot constrained trajectory if available.
    // (If fitted track is outside HT array, it it put in the closest bin inside it).
    std::pair<unsigned int, unsigned int> cell(const L1fittedTrack* fitTrk) const override;

    // Check if specified cell has been merged with its 2x2 neighbours into a single cell,
    // as it is in low Pt region.
    bool mergedCell(unsigned int iQoverPtBin, unsigned int jPhiTrkBin) const;

    // Number of stubs received from GP, irrespective of whether the stub was actually stored in
    // a cell in the HT array.
    unsigned int nReceivedStubs() const { return nReceivedStubs_; }

    // Determine which m-bin (q/pt) range the specified track is in. (Used if outputting each m bin range on a different opto-link).
    unsigned int getMbinRange(const L1track2D& trk) const;

  private:
    // For a given Q/Pt bin, find the range of phi bins that a given stub is consistent with.
    std::pair<unsigned int, unsigned int> iPhiRange(const Stub* stub,
                                                    unsigned int iQoverPtBin,
                                                    bool debug = false) const;

    // Check that limitations of firmware would not prevent stub being stored correctly in this HT column.
    void countFirmwareErrors(unsigned int iQoverPtBin,
                             unsigned int iPhiTrkBinMin,
                             unsigned int iPhiTrkBinMax,
                             unsigned int jPhiTrkBinMinLast,
                             unsigned int jPhiTrkBinMaxLast);

    // Calculate line |gradient| of stubs in HT array, so can check it doesn't exceed 1.
    float calcLineGradArray(float r) const;

    // If requested, kill those tracks in this sector that can't be read out during the time-multiplexed period, because
    // the HT has associated too many stubs to tracks.
    std::list<L1track2D> killTracksBusySec(const std::list<L1track2D>& tracks) const override;

    // Define the order in which the hardware processes rows of the HT array when it outputs track candidates.
    // Currently corresponds to highest Pt tracks first.
    // If two tracks have the same Pt, the -ve charge one is output before the +ve charge one.
    std::vector<unsigned int> rowOrder(unsigned int numRows) const override;

  private:
    float invPtToDphi_;  // conversion constant.

    HTshape shape_;
    std::vector<std::vector<std::pair<float, float> > > cellCenters_;

    //--- Specifications of HT array.

    float maxAbsQoverPtAxis_;        // Max. |q/Pt| covered by  HT array.
    unsigned int nBinsQoverPtAxis_;  // Number of bins in HT array in q/Pt.
    float binSizeQoverPtAxis_;       // HT array bin size in q/Pt.

    float chosenRofPhi_;            // Use phi of track at radius="chosenRofPhi" to define one of the r-phi HT axes.
    float phiCentreSector_;         // phiTrk angle of centre of this (eta,phi) sector.
    float maxAbsPhiTrkAxis_;        // Half-width of phiTrk axis in HT array.
    unsigned int nBinsPhiTrkAxis_;  // Number of bins in HT array in phiTrk axis.
    float binSizePhiTrkAxis_;       // HT array bin size in phiTrk
    // Optionally merge 2x2 neighbouring cells into a single cell at low Pt, to reduce efficiency loss due to scattering. (Used also by mini-HT).
    bool enableMerge2x2_;
    float minInvPtToMerge2x2_;

    //--- Options when filling HT array.

    // Take all cells in HT array crossed by line corresponding to each stub (= 0) or take only some to reduce rate at cost of efficiency ( > 0)
    unsigned int killSomeHTCellsRphi_;
    // Options for killing stubs/tracks that cant be sent within time-multiplexed period.
    bool busyInputSectorKill_;
    bool busySectorKill_;
    unsigned int busyInputSectorNumStubs_;
    unsigned int busySectorNumStubs_;
    std::vector<unsigned int> busySectorMbinRanges_;
    std::vector<unsigned int> busySectorMbinOrder_;
    bool busySectorUseMbinRanges_;
    bool busySectorUseMbinOrder_;
    std::vector<unsigned int> busySectorMbinLow_;
    std::vector<unsigned int> busySectorMbinHigh_;

    // Number of stubs received from GP, irrespective of whether the stub was actually stored in
    // a cell in the HT array.
    unsigned int nReceivedStubs_;

    // Error monitoring.
    ErrorMonitor* errMon_;

    // ... The Hough transform array data is in the base class ...

    // ... The list of found track candidates is in the base class ...
  };

}  // namespace tmtt

#endif
