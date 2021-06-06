#ifndef L1Trigger_TrackFindingTMTT_MuxHToutputs_h
#define L1Trigger_TrackFindingTMTT_MuxHToutputs_h

#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"
#include "L1Trigger/TrackFindingTMTT/interface/Array2D.h"

#include <vector>
#include <memory>

//==================================================================================================
/**
* Multiplex the tracks found by several HT onto a single output optical link.
* (where throughout this class, the word "link" corresponds to a pair of links in the hardware).
* so that tracks that can't be sent down the link within the time-multiplexed period are killed.
*
* This class replaces the 2D track collection in the r-phi HTs with the subset of the tracks
* that can be output within the TM period.
*
* If you wish to change the multiplexing algorithm, then edit this class ...
*/
//==================================================================================================

namespace tmtt {

  class Settings;

  class MuxHToutputs {
  public:
    enum class MuxAlgoName { None = 0, mBinPerLink = 1 };

    // Initialize constants from configuration parameters.
    MuxHToutputs(const Settings* settings);

    // Determine which tracks are transmitted on each HT output optical link, taking into account the multiplexing
    // of multiple (eta,phi) sectors onto single links and the truncation of the tracks caused by the requirement
    // to output all the tracks within the time-multiplexed period.
    // This function replaces the 2D track collection in the r-phi HT with the subset surviving the TM cut.
    void exec(Array2D<std::unique_ptr<HTrphi>>& mHtRphis) const;

    // Determine number of optical links used to output tracks from each phi nonant
    // (where "link" refers to a pair of links in the hardware).
    unsigned int numLinksPerNonant() const {
      unsigned int iCorr = (settings_->miniHTstage()) ? 1 : 0;
      return numPhiSecPerNon_ * numEtaRegions_ * (busySectorMbinRanges_.size() - iCorr) / this->muxFactor();
    }

  private:
    // Define the number of (eta,phi) sectors that each output opto-link takes tracks from. (Depends on MUX scheme).
    unsigned int muxFactor() const;

    // Define the MUX algorithm by which tracks from the specified m-bin range in the HT for a given (phi,eta)
    // sector within a phi nonant are multiplexed onto a single output optical link.
    unsigned int linkID(unsigned int iSecInNon, unsigned int iEtaReg, unsigned int mBinRange) const;

    // Do sanity check of the MUX algorithm implemented in linkID().
    void sanityCheck();

  private:
    const Settings* settings_;  // Configuration parameters

    // Configuration parameters
    MuxAlgoName muxOutputsHT_;
    unsigned int numPhiNonants_;
    unsigned int numPhiSectors_;
    unsigned int numPhiSecPerNon_;
    unsigned int numEtaRegions_;
    bool busySectorKill_;
    unsigned int busySectorNumStubs_;
    std::vector<unsigned int> busySectorMbinRanges_;
    bool busySectorUseMbinRanges_;
  };

}  // namespace tmtt

#endif
