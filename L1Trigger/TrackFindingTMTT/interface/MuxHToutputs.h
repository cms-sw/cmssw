#ifndef __MUXHTOUTPUTS_H__
#define __MUXHTOUTPUTS_H__

#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"

#include "boost/numeric/ublas/matrix.hpp"
#include <vector>

using namespace std;
using boost::numeric::ublas::matrix;

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

namespace TMTT {

class Settings;

class MuxHToutputs {

public:

  // Initialize constants from configuration parameters.
  MuxHToutputs(const Settings* settings);

  ~MuxHToutputs() {}

  // Determine which tracks are transmitted on each HT output optical link, taking into account the multiplexing
  // of multiple (eta,phi) sectors onto single links and the truncation of the tracks caused by the requirement
  // to output all the tracks within the time-multiplexed period.
  // This function replaces the 2D track collection in the r-phi HT with the subset surviving the TM cut.
  void exec(matrix<HTrphi>& mHtRphis) const;

  // Determine number of optical links used to output tracks from each phi nonant
  // (where "link" refers to a pair of links in the hardware).
  unsigned int numLinksPerNonant() const {unsigned int iCorr = (settings_->miniHTstage()) ? 1 : 0; return numPhiSecPerNon_ * numEtaRegions_ * (busySectorMbinRanges_.size() - iCorr)/ this->muxFactor();}

private:

  // Define the number of (eta,phi) sectors that each output opto-link takes tracks from. (Depends on MUX scheme).
  unsigned int muxFactor() const;

  // Define the MUX algorithm by which tracks from the specified m-bin range in the HT for a given (phi,eta)
  // sector within a phi nonant are multiplexed onto a single output optical link.
  unsigned int linkID(unsigned int iSecInNon, unsigned int iEtaReg, unsigned int mBinRange) const;

  // Do sanity check of the MUX algorithm implemented in linkID().
  void sanityCheck();

private:

  const Settings* settings_; // Configuration parameters

  // Configuration parameters
  unsigned int         muxOutputsHT_;
  unsigned int         numPhiNonants_;
  unsigned int         numPhiSectors_;
  unsigned int         numPhiSecPerNon_;
  unsigned int         numEtaRegions_;
  bool                 busySectorKill_;
  unsigned int         busySectorNumStubs_;
  vector<unsigned int> busySectorMbinRanges_;
  bool                 busySectorUseMbinRanges_;
};

}

#endif

