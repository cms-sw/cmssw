//--- Note that the word "link" appearing in the C++ or comments in this class actually corresponds
//--- to a pair of links in the hardware.

#include "L1Trigger/TrackFindingTMTT/interface/MuxHToutputs.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/PrintL1trk.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <sstream>
#include <mutex>

using namespace std;

namespace tmtt {

  //=== Initialize constants from configuration parameters.

  MuxHToutputs::MuxHToutputs(const Settings* settings)
      : settings_(settings),
        muxOutputsHT_(static_cast<MuxAlgoName>(settings_->muxOutputsHT())),
        numPhiNonants_(settings_->numPhiNonants()),
        numPhiSectors_(settings_->numPhiSectors()),
        numPhiSecPerNon_(numPhiSectors_ / numPhiNonants_),
        numEtaRegions_(settings_->numEtaRegions()),
        busySectorKill_(settings_->busySectorKill()),          // Kill excess tracks flowing out of HT?
        busySectorNumStubs_(settings_->busySectorNumStubs()),  // Max. num. of stubs that can be sent within TM period
        busySectorMbinRanges_(
            settings_->busySectorMbinRanges()),  // Individual m bin (=q/Pt) ranges to be output to opto-links.
        busySectorUseMbinRanges_(not busySectorMbinRanges_.empty())  // m bin ranges option disabled if vector empty.
  {
    // Implemented MUX algorithm relies on same number of sectors per nonant.
    if (numPhiSectors_ % numPhiNonants_ != 0)
      throw cms::Exception("BadConfig")
          << "MuxHToutputs: Number of phi sectors is not a multiple of number of nonants!";

    if (!busySectorUseMbinRanges_)
      throw cms::Exception("BadConfig") << "MuxHToutputs: The implemented MUX algorithm requires you to be using the "
                                           "busySectorMbinRanges cfg option!";

    // Check that the MUX algorithm implemented in linkID() is not obviously wrong.
    this->sanityCheck();

    std::stringstream text;
    text << "=== The R-PHI HT output is multiplexed onto " << this->numLinksPerNonant()
         << " pairs of opto-links per nonant.";
    static std::once_flag printOnce;
    std::call_once(
        printOnce, [](string t) { PrintL1trk() << t; }, text.str());
  }

  //=== Determine which tracks are transmitted on each HT output optical link, taking into account the multiplexing
  //=== of multiple (eta,phi) sectors onto single links and the truncation of the tracks caused by the requirement
  //=== to output all the tracks within the time-multiplexed period.
  //=== This function replaces the 2D track collection in the r-phi HT with the subset surviving the TM cut.

  void MuxHToutputs::exec(Array2D<unique_ptr<HTrphi>>& mHtRphis) const {
    // As this loops over sectors in order of increasing sector number, this MUX algorithm always transmits tracks
    // from the lowest sector numbers on each link first. So the highest sector numbers are more likely to be
    // truncated by the TM period. The algorithm assumes that two or more m-bin ranges from the same sector will never
    // be transmitted down the same link, as if this happens, it does not predict the order in which they will be
    // transmitted.

    for (unsigned int iPhiNon = 0; iPhiNon < numPhiNonants_; iPhiNon++) {
      vector<unsigned int> numStubsPerLink(this->numLinksPerNonant(), 0);

      for (unsigned int iSecInNon = 0; iSecInNon < numPhiSecPerNon_; iSecInNon++) {
        unsigned int iPhiSec = iPhiNon * numPhiSecPerNon_ + iSecInNon;

        for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
          HTrphi* htRphi = mHtRphis(iPhiSec, iEtaReg).get();  // Get a mutable version of the r-phi HT.

          list<L1track2D> keptTracks;
          const list<L1track2D>& tracks = htRphi->trackCands2D();

          for (const L1track2D& trk : tracks) {
            L1track2D trkTmp = trk;
            unsigned int nStubs = trkTmp.numStubs();                // #stubs on this track.
            unsigned int mBinRange = htRphi->getMbinRange(trkTmp);  // Which m bin range is this track in?
            // Get the output optical link corresponding to this sector & m-bin range.
            unsigned int link = this->linkID(iSecInNon, iEtaReg, mBinRange);
            // Make a note of opto-link number inside track object.
            trkTmp.setOptoLinkID(link);

            numStubsPerLink[link] += nStubs;
            // Check if this track can be output within the time-multiplexed period.
            bool keep = ((not busySectorKill_) || (numStubsPerLink[link] <= busySectorNumStubs_));
            // FIX: with 2 GeV threshold, this causes significant truncation.
            // Consider using one output link for each phi sector in nonant
            if (keep)
              keptTracks.push_back(trkTmp);
          }

          // Replace the collection of 2D tracks in the r-phi HT with the subset of them surviving the TM cut.
          htRphi->replaceTrackCands2D(keptTracks);
        }
      }
    }
  }

  //=== Define the number of (eta,phi) sectors that each output opto-link takes tracks from. (Depends on MUX scheme).

  unsigned int MuxHToutputs::muxFactor() const {
    if (muxOutputsHT_ == MuxAlgoName::mBinPerLink) {
      return numEtaRegions_ * numPhiSecPerNon_;
    } else {
      throw cms::Exception("BadConfig") << "MuxHToutputs: Unknown MuxOutputsHT configuration option!";
    }
  }

  //=== Define the MUX algorithm by which tracks from the specified m-bin range in the HT for a given (phi,eta)
  //=== sector within a phi nonant are multiplexed onto a single output optical link.

  unsigned int MuxHToutputs::linkID(unsigned int iSecInNon, unsigned int iEtaReg, unsigned int mBinRange) const {
    unsigned int link;

    if (muxOutputsHT_ == MuxAlgoName::mBinPerLink) {
      //--- This is the Sept. 2019 Mux for the transverse HT readout organised by m-bin. (Each m bin in entire nonant goes to a different link).

      link = 0;
      link += mBinRange;

    } else {
      throw cms::Exception("BadConfig") << "MuxHToutputs: Unknown MuxOutputsHT configuration option!";
    }

    if (link >= this->numLinksPerNonant())
      throw cms::Exception("LogicError") << "MuxHToutputs: Calculated link ID exceeded expected number of links! "
                                         << link << " " << this->numLinksPerNonant();
    return link;
  }

  //=== Do sanity check of the MUX algorithm implemented in linkID().

  void MuxHToutputs::sanityCheck() {
    if (numPhiSecPerNon_ * numEtaRegions_ % this->muxFactor() != 0)
      throw cms::Exception("LogicError")
          << "MuxHToutputs: Number of sectors per phi nonant is not a multiple of muxFactor().";

    vector<unsigned int> nObsElementsPerLink(this->numLinksPerNonant(), 0);
    for (unsigned int iSecInNon = 0; iSecInNon < numPhiSecPerNon_; iSecInNon++) {
      for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
        unsigned int iCorr = (settings_->miniHTstage()) ? 1 : 0;
        for (unsigned int mBinRange = 0; mBinRange < busySectorMbinRanges_.size() - iCorr; mBinRange++) {
          unsigned int link = this->linkID(iSecInNon, iEtaReg, mBinRange);
          nObsElementsPerLink[link] += 1;
        }
      }
    }
  }

}  // namespace tmtt
