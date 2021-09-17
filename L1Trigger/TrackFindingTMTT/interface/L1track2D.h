#ifndef L1Trigger_TrackFindingTMTT_L1track2D_h
#define L1Trigger_TrackFindingTMTT_L1track2D_h

#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1trackBase.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"

#include <vector>
#include <utility>

//=== L1 track cand found in 2 dimensions.
//=== Gives access to all stubs on track and to its 2D helix parameters.
//=== Also calculates & gives access to associated truth particle (Tracking Particle) if any.

namespace tmtt {

  class L1track2D : public L1trackBase {
  public:
    // Give stubs on track, its cell location inside HT array, its 2D helix parameters.
    L1track2D(const Settings* settings,
              const std::vector<Stub*>& stubs,
              std::pair<unsigned int, unsigned int> cellLocationHT,
              std::pair<float, float> helix2D,
              unsigned int iPhiSec,
              unsigned int iEtaReg,
              unsigned int optoLinkID,
              bool mergedHTcell)
        : L1trackBase(),
          settings_(settings),
          stubs_(stubs),
          stubsConst_(stubs_.begin(), stubs_.end()),
          cellLocationHT_(cellLocationHT),
          helix2D_(helix2D),
          estValid_(false),
          estZ0_(0.),
          estTanLambda_(0.),
          iPhiSec_(iPhiSec),
          iEtaReg_(iEtaReg),
          optoLinkID_(optoLinkID),
          mergedHTcell_(mergedHTcell) {
      nLayers_ = Utility::countLayers(settings, stubs_);  // Count tracker layers these stubs are in
      matchedTP_ = Utility::matchingTP(settings,
                                       stubs_,
                                       nMatchedLayers_,
                                       matchedStubs_);  // Find associated truth particle & calculate info about match.
    }

    ~L1track2D() override = default;

    //--- Get information about the reconstructed track.

    // Get stubs on track candidate.
    const std::vector<const Stub*>& stubsConst() const override { return stubsConst_; }
    const std::vector<Stub*>& stubs() const override { return stubs_; }
    // Get number of stubs on track candidate.
    unsigned int numStubs() const override { return stubs_.size(); }
    // Get number of tracker layers these stubs are in.
    unsigned int numLayers() const override { return nLayers_; }
    // Get cell location of track candidate in Hough Transform array in units of bin number.
    std::pair<unsigned int, unsigned int> cellLocationHT() const override { return cellLocationHT_; }
    // The two conventionally agreed track helix parameters relevant in this 2D plane.
    // i.e. (q/Pt, phi0).
    std::pair<float, float> helix2D() const { return helix2D_; }

    //--- User-friendlier access to the helix parameters obtained from track location inside HT array.

    float qOverPt() const override { return helix2D_.first; }
    float phi0() const override { return helix2D_.second; }

    //--- In the case of tracks found by the r-phi HT, a rough estimate of the (z0, tan_lambda) may be provided by any r-z
    //--- track filter run after the r-phi HT. These two functions give std::set/get access to these.
    //--- The "get" function returns a boolean indicating if an estimate exists (i.e. "set" has been called).

    void setTrkEstZ0andTanLam(float estZ0, float estTanLambda) {
      estZ0_ = estZ0;
      estTanLambda_ = estTanLambda;
      estValid_ = true;
    }

    bool trkEstZ0andTanLam(float& estZ0, float& estTanLambda) const {
      estZ0 = estZ0_;
      estTanLambda = estTanLambda_;
      return estValid_;
    }

    //--- Get phi sector and eta region used by track finding code that this track is in.
    unsigned int iPhiSec() const override { return iPhiSec_; }
    unsigned int iEtaReg() const override { return iEtaReg_; }

    //--- Opto-link ID used to send this track from HT to Track Fitter. Both read & write functions.
    unsigned int optoLinkID() const override { return optoLinkID_; }
    void setOptoLinkID(unsigned int linkID) { optoLinkID_ = linkID; }

    //--- Was this track produced from a marged HT cell (e.g. 2x2)?
    bool mergedHTcell() const { return mergedHTcell_; }

    //--- Get information about its association (if any) to a truth Tracking Particle.

    // Get matching tracking particle (=nullptr if none).
    const TP* matchedTP() const override { return matchedTP_; }
    // Get the matched stubs.
    const std::vector<const Stub*>& matchedStubs() const override { return matchedStubs_; }
    // Get number of matched stubs.
    unsigned int numMatchedStubs() const override { return matchedStubs_.size(); }
    // Get number of tracker layers with matched stubs.
    unsigned int numMatchedLayers() const override { return nMatchedLayers_; }

  private:
    //--- Configuration parameters
    const Settings* settings_;

    //--- Information about the reconstructed track from Hough transform.
    std::vector<Stub*> stubs_;
    std::vector<const Stub*> stubsConst_;
    unsigned int nLayers_;
    std::pair<unsigned int, unsigned int> cellLocationHT_;
    std::pair<float, float> helix2D_;

    //--- Rough estimate of r-z track parameters from r-z filter, which may be present in case of r-phi Hough transform
    bool estValid_;
    float estZ0_;
    float estTanLambda_;

    unsigned int iPhiSec_;
    unsigned int iEtaReg_;
    unsigned int optoLinkID_;

    bool mergedHTcell_;

    //--- Information about its association (if any) to a truth Tracking Particle.
    const TP* matchedTP_;
    std::vector<const Stub*> matchedStubs_;
    unsigned int nMatchedLayers_;
  };

}  // namespace tmtt

#endif
