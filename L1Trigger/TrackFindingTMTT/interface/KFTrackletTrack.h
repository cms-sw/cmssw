#ifndef L1Trigger_TrackFindingTMTT_KFTrackletTrack_h
#define L1Trigger_TrackFindingTMTT_KFTrackletTrack_h

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1trackBase.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/DigitalTrack.h"

#include <vector>
#include <utility>
#include <string>

//=== This is used uniquely for HYBRID TRACKING.
//=== It is the equivalent of class L1fittedTrack.
//===
//=== This represents a fitted L1 track candidate found in 3 dimensions.
//=== It gives access to the fitted helix parameters & chi2 etc.
//=== It also calculates & gives access to associated truth particle (Tracking Particle) if any.
//=== It also gives access to the 3D hough-transform track candidate (L1track3D) on which the fit was run.

namespace tmtt {

  class KFTrackletTrack {
  public:
    // Store a new fitted track, specifying the input Hough transform track, the stubs used for the fit,
    // bit-encoded hit layers,
    // the fitted helix parameters & chi2,
    // and the number of helix parameters being fitted (=5 if d0 is fitted, or =4 if d0 is not fitted).
    // Also specify phi sector and eta region used by track-finding code that this track was in.
    // And if track fit declared this to be a valid track (enough stubs left on track after fit etc.).
    KFTrackletTrack(const L1track3D* l1track3D,
                    const std::vector<const Stub*>& stubs,
                    unsigned int hitPattern,
                    float qOverPt,
                    float d0,
                    float phi0,
                    float z0,
                    float tanLambda,
                    float chi2rphi,
                    float chi2rz,
                    unsigned int nHelixParam,
                    unsigned int iPhiSec,
                    unsigned int iEtaReg,
                    bool accepted = true,
                    bool done_bcon = false,
                    float qOverPt_bcon = 0.,
                    float d0_bcon = 0.,
                    float phi0_bcon = 0.,
                    float chi2rphi_bcon = 0.)
        : l1track3D_(l1track3D),
          stubs_(stubs),
          hitPattern_(hitPattern),
          qOverPt_(qOverPt),
          d0_(d0),
          phi0_(phi0),
          z0_(z0),
          tanLambda_(tanLambda),
          chi2rphi_(chi2rphi),
          chi2rz_(chi2rz),
          done_bcon_(done_bcon),
          qOverPt_bcon_(qOverPt_bcon),
          d0_bcon_(d0_bcon),
          phi0_bcon_(phi0_bcon),
          chi2rphi_bcon_(chi2rphi_bcon),
          nHelixParam_(nHelixParam),
          iPhiSec_(iPhiSec),
          iEtaReg_(iEtaReg),
          optoLinkID_(l1track3D->optoLinkID()),
          nSkippedLayers_(0),
          numUpdateCalls_(0),
          numIterations_(0),
          accepted_(accepted) {}

    //--- Set/get additional info about fitted track that is specific to individual track fit algorithms (KF, LR, chi2)
    //--- and is used for debugging/histogramming purposes.

    void setInfoKF(unsigned int nSkippedLayers, unsigned int numUpdateCalls) {
      nSkippedLayers_ = nSkippedLayers;
      numUpdateCalls_ = numUpdateCalls;
    }

    void infoKF(unsigned int& nSkippedLayers, unsigned int& numUpdateCalls) const {
      nSkippedLayers = nSkippedLayers_;
      numUpdateCalls = numUpdateCalls_;
    }

    const L1track3D* l1track3D() const { return l1track3D_; }

    // Get stubs on fitted track (can differ from those on HT track if track fit kicked out stubs with bad residuals)
    const std::vector<const Stub*>& stubs() const { return stubs_; }
    // Get number of stubs on fitted track.
    unsigned int numStubs() const { return stubs_.size(); }
    // Get number of tracker layers these stubs are in.
    unsigned int numLayers() const { return nLayers_; }
    // Get number of stubs deleted from track candidate by fitter (because they had large residuals)
    unsigned int numKilledStubs() const { return l1track3D_->numStubs() - this->numStubs(); }

    // Get bit-encoded hit pattern (where layer number assigned by increasing distance from origin, according to layers track expected to cross).
    unsigned int hitPattern() const { return hitPattern_; }

    //--- Get the fitted track helix parameters.

    float qOverPt() const { return qOverPt_; }
    float charge() const { return (qOverPt_ > 0 ? 1 : -1); }
    float invPt() const { return std::abs(qOverPt_); }
    // Protect pt against 1/pt = 0.
    float pt() const {
      constexpr float small = 1.0e-6;
      return 1. / (small + this->invPt());
    }
    float d0() const { return d0_; }
    float phi0() const { return phi0_; }
    float z0() const { return z0_; }
    float tanLambda() const { return tanLambda_; }
    float theta() const { return atan2(1., tanLambda_); }  // Use atan2 to ensure 0 < theta < pi.
    float eta() const { return -log(tan(0.5 * this->theta())); }

    //--- Get the fitted helix parameters with beam-spot constraint.
    //--- If constraint not applied (e.g. 4 param fit) then these are identical to unconstrained values.

    bool done_bcon() const { return done_bcon_; }  // Was beam-spot constraint aplied?
    float qOverPt_bcon() const { return qOverPt_bcon_; }
    float charge_bcon() const { return (qOverPt_bcon_ > 0 ? 1 : -1); }
    float invPt_bcon() const { return std::abs(qOverPt_bcon_); }
    float pt_bcon() const { return 1. / (1.0e-6 + this->invPt_bcon()); }
    float phi0_bcon() const { return phi0_bcon_; }
    float d0_bcon() const { return d0_bcon_; }

    // Phi and z coordinates at which track crosses "chosenR" values used by r-phi HT and rapidity sectors respectively.
    // (Optionally with beam-spot constraint applied).
    float phiAtChosenR(bool beamConstraint) const {
      if (beamConstraint) {
        return reco::deltaPhi(phi0_bcon_ -
                                  asin((settings_->invPtToDphi() * settings_->chosenRofPhi()) * qOverPt_bcon_) -
                                  d0_bcon_ / (settings_->chosenRofPhi()),
                              0.);
      } else {
        return reco::deltaPhi(phi0_ - asin((settings_->invPtToDphi() * settings_->chosenRofPhi()) * qOverPt_) -
                                  d0_ / (settings_->chosenRofPhi()),
                              0.);
      }
    }
    float zAtChosenR() const {
      return (z0_ + (settings_->chosenRofZ()) * tanLambda_);
    }  // neglects transverse impact parameter & track curvature.

    // Get the number of helix parameters being fitted (=5 if d0 is fitted or =4 if d0 is not fitted).
    float nHelixParam() const { return nHelixParam_; }

    // Get the fit degrees of freedom, chi2 & chi2/DOF
    unsigned int numDOF() const { return 2 * this->numStubs() - nHelixParam_; }
    unsigned int numDOFrphi() const { return this->numStubs() - (nHelixParam_ - 2); }
    unsigned int numDOFrz() const { return this->numStubs() - 2; }
    float chi2rphi() const { return chi2rphi_; }
    float chi2rz() const { return chi2rz_; }
    float chi2() const { return chi2rphi_ + chi2rz_; }
    float chi2dof() const { return (this->chi2()) / this->numDOF(); }

    //--- Ditto, but if beam-spot constraint is applied.
    //--- If constraint not applied (e.g. 4 param fit) then these are identical to unconstrained values.
    unsigned int numDOF_bcon() const { return (this->numDOF() - 1); }
    unsigned int numDOFrphi_bcon() const { return (this->numDOFrphi() - 1); }
    float chi2rphi_bcon() const { return chi2rphi_bcon_; }
    float chi2_bcon() const { return chi2rphi_bcon_ + chi2rz_; }
    float chi2dof_bcon() const { return (this->chi2_bcon()) / this->numDOF_bcon(); }

    //--- Get phi sector and eta region used by track finding code that this track is in.
    unsigned int iPhiSec() const { return iPhiSec_; }
    unsigned int iEtaReg() const { return iEtaReg_; }

    //--- Opto-link ID used to send this track from HT to Track Fitter
    unsigned int optoLinkID() const { return optoLinkID_; }

    //--- Get whether the track has been rejected or accepted by the fit

    bool accepted() const { return accepted_; }

    // Digitize track and degrade helix parameter resolution according to effect of digitisation.
    void digitizeTrack(const std::string& fitterName);

    // Access to detailed info about digitized track
    const DigitalTrack* digitaltrack() const { return digitalTrack_.get(); }

  private:
    //--- Configuration parameters
    const Settings* settings_;

    //--- The 3D hough-transform track candidate which was fitted.
    const L1track3D* l1track3D_;

    //--- The stubs on the fitted track (can differ from those on HT track if fit kicked off stubs with bad residuals)
    std::vector<const Stub*> stubs_;
    unsigned int nLayers_;

    //--- Bit-encoded hit pattern (where layer number assigned by increasing distance from origin, according to layers track expected to cross).
    unsigned int hitPattern_;

    //--- The fitted helix parameters and fit chi-squared.
    float qOverPt_;
    float d0_;
    float phi0_;
    float z0_;
    float tanLambda_;
    float chi2rphi_;
    float chi2rz_;

    //--- Ditto with beam-spot constraint applied in case of 5-parameter fit, plus boolean to indicate
    bool done_bcon_;
    float qOverPt_bcon_;
    float d0_bcon_;
    float phi0_bcon_;
    float chi2rphi_bcon_;

    //--- The number of helix parameters being fitted (=5 if d0 is fitted or =4 if d0 is not fitted).
    unsigned int nHelixParam_;

    //--- Phi sector and eta region used track finding code that this track was in.
    unsigned int iPhiSec_;
    unsigned int iEtaReg_;
    //--- Opto-link ID from HT to Track Fitter.
    unsigned int optoLinkID_;

    //--- Information about its association (if any) to a truth Tracking Particle.
    const TP* matchedTP_;
    std::vector<const Stub*> matchedStubs_;
    unsigned int nMatchedLayers_;

    //--- Info specific to KF fitter.
    unsigned int nSkippedLayers_;
    unsigned int numUpdateCalls_;
    //--- Info specific to LR fitter.
    int numIterations_;
    std::string lostMatchingState_;
    std::unordered_map<std::string, int> stateCalls_;

    std::shared_ptr<DigitalTrack> digitalTrack_;  // Class used to digitize track if required.

    //--- Has the track fit declared this to be a valid track?
    bool accepted_;
  };

}  // namespace tmtt

#endif
