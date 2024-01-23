#ifndef L1Trigger_TrackFindingTMTT_L1fittedTrack_h
#define L1Trigger_TrackFindingTMTT_L1fittedTrack_h

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1trackBase.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/Sector.h"
#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"
#include "L1Trigger/TrackFindingTMTT/interface/DigitalTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFTrackletTrack.h"

#include <vector>
#include <set>
#include <utility>
#include <string>
#include <memory>

//=== This represents a fitted L1 track candidate found in 3 dimensions.
//=== It gives access to the fitted helix parameters & chi2 etc.
//=== It also calculates & gives access to associated truth particle (Tracking Particle) if any.
//=== It also gives access to the 3D hough-transform track candidate (L1track3D) on which the fit was run.

namespace tmtt {

  class L1fittedTrack : public L1trackBase {
  public:
    // Store a new fitted track, specifying the input Hough transform track, the stubs used for the fit,
    // bit-encoded hit layer pattern (numbered by increasing distance from origin),
    // the fitted helix parameters & chi2,
    // and the number of helix parameters being fitted (=5 if d0 is fitted, or =4 if d0 is not fitted).
    // And if track fit declared this to be a valid track (enough stubs left on track after fit etc.).
    L1fittedTrack(const Settings* settings,
                  const L1track3D* l1track3D,
                  const std::vector<Stub*>& stubs,
                  unsigned int hitPattern,
                  float qOverPt,
                  float d0,
                  float phi0,
                  float z0,
                  float tanLambda,
                  float chi2rphi,
                  float chi2rz,
                  unsigned int nHelixParam,
                  bool accepted = true)
        : L1trackBase(),
          settings_(settings),
          l1track3D_(l1track3D),
          stubs_(stubs),
          stubsConst_(stubs_.begin(), stubs_.end()),
          hitPattern_(hitPattern),
          qOverPt_(qOverPt),
          d0_(d0),
          phi0_(phi0),
          z0_(z0),
          tanLambda_(tanLambda),
          chi2rphi_(chi2rphi),
          chi2rz_(chi2rz),
          done_bcon_(false),
          qOverPt_bcon_(qOverPt),
          d0_bcon_(d0),
          phi0_bcon_(phi0),
          chi2rphi_bcon_(chi2rphi),
          nHelixParam_(nHelixParam),
          nSkippedLayers_(0),
          numUpdateCalls_(0),
          numIterations_(0),
          accepted_(accepted) {
      if (l1track3D != nullptr) {
        iPhiSec_ = l1track3D->iPhiSec();
        iEtaReg_ = l1track3D->iEtaReg();
        optoLinkID_ = l1track3D->optoLinkID();
      } else {  // Rejected track
        iPhiSec_ = 0;
        iEtaReg_ = 0;
        optoLinkID_ = 0;
      }
      if (settings != nullptr) {
        // Count tracker layers these stubs are in
        nLayers_ = Utility::countLayers(settings, stubs_);
        // Find associated truth particle & calculate info about match.
        matchedTP_ = Utility::matchingTP(settings, stubs_, nMatchedLayers_, matchedStubs_);
      } else {  // Rejected track
        nLayers_ = 0;
        matchedTP_ = nullptr;
      }
      // Set d0 = 0 for 4 param fit, in case fitter didn't do it.
      if (nHelixParam == 4) {
        d0_ = 0.;
        d0_bcon_ = 0.;
      }
      if (settings != nullptr && not settings->hybrid()) {
        //Sector class used to check if fitted track trajectory is in expected sector.
        secTmp_ = std::make_shared<Sector>(settings, iPhiSec_, iEtaReg_);
        // HT class used to identify HT cell that corresponds to fitted helix parameters.
        htRphiTmp_ = std::make_shared<HTrphi>(
            settings, iPhiSec_, iEtaReg_, secTmp_->etaMin(), secTmp_->etaMax(), secTmp_->phiCentre());
        this->setConsistentHTcell();
      } else {
        consistentCell_ = false;
      }
    }

    // Creates track rejected by fitter.
    L1fittedTrack() : L1fittedTrack(nullptr, nullptr, noStubs_, 0, 0., 0., 0., 0., 0., 0., 0., 0, false) {}

    ~L1fittedTrack() override = default;

    //--- Optionally std::set track helix params & chi2 if beam-spot constraint is used (for 5-parameter fit).
    void setBeamConstr(float qOverPt_bcon, float phi0_bcon, float chi2rphi_bcon, bool accepted) {
      done_bcon_ = true;
      qOverPt_bcon_ = qOverPt_bcon;
      d0_bcon_ = 0.0, phi0_bcon_ = phi0_bcon;
      chi2rphi_bcon_ = chi2rphi_bcon;
      accepted_ = accepted;
    }

    //--- Set/get additional info about fitted track that is specific to individual track fit algorithms (KF, LR, chi2)
    //--- and is used for debugging/histogramming purposes.

    void setInfoKF(unsigned int nSkippedLayers, unsigned int numUpdateCalls) {
      nSkippedLayers_ = nSkippedLayers;
      numUpdateCalls_ = numUpdateCalls;
    }
    void setInfoLR(int numIterations, std::string lostMatchingState, std::unordered_map<std::string, int> stateCalls) {
      numIterations_ = numIterations;
      lostMatchingState_ = lostMatchingState;
      stateCalls_ = stateCalls;
    }
    void setInfoCHI2() {}

    void infoKF(unsigned int& nSkippedLayers, unsigned int& numUpdateCalls) const {
      nSkippedLayers = nSkippedLayers_;
      numUpdateCalls = numUpdateCalls_;
    }
    void infoLR(int& numIterations,
                std::string& lostMatchingState,
                std::unordered_map<std::string, int>& stateCalls) const {
      numIterations = numIterations_;
      lostMatchingState = lostMatchingState_;
      stateCalls = stateCalls_;
    }
    void infoCHI2() const {}

    //--- Convert fitted track to KFTrackletTrack format, for use with HYBRID.

    KFTrackletTrack returnKFTrackletTrack() {
      KFTrackletTrack trk_(l1track3D(),
                           stubsConst(),
                           hitPattern(),
                           qOverPt(),
                           d0(),
                           phi0(),
                           z0(),
                           tanLambda(),
                           chi2rphi(),
                           chi2rz(),
                           nHelixParam(),
                           iPhiSec(),
                           iEtaReg(),
                           accepted(),
                           done_bcon(),
                           qOverPt_bcon(),
                           d0_bcon(),
                           phi0_bcon(),
                           chi2rphi_bcon());
      return trk_;
    }

    //--- Get the 3D Hough transform track candididate corresponding to the fitted track,
    //--- Provide direct access to some of the info it contains.

    // Get track candidate from HT (before fit).
    const L1track3D* l1track3D() const { return l1track3D_; }

    // Get stubs on fitted track (can differ from those on HT track if track fit kicked out stubs with bad residuals)
    const std::vector<const Stub*>& stubsConst() const override { return stubsConst_; }
    const std::vector<Stub*>& stubs() const override { return stubs_; }
    // Get number of stubs on fitted track.
    unsigned int numStubs() const override { return stubs_.size(); }
    // Get number of tracker layers these stubs are in.
    unsigned int numLayers() const override { return nLayers_; }
    // Get number of stubs deleted from track candidate by fitter (because they had large residuals)
    unsigned int numKilledStubs() const { return l1track3D_->numStubs() - this->numStubs(); }
    // Get bit-encoded hit pattern (where layer number assigned by increasing distance from origin, according to layers track expected to cross).
    unsigned int hitPattern() const { return hitPattern_; }

    // Get Hough transform cell locations in units of bin number, corresponding to the fitted helix parameters of the track.
    // Always uses the beam-spot constrained helix params if they are available.
    // (If fitted track is outside HT array, it it put in the closest bin inside it).
    std::pair<unsigned int, unsigned int> cellLocationFit() const { return htRphiTmp_->cell(this); }
    // Also get HT cell determined by Hough transform.
    std::pair<unsigned int, unsigned int> cellLocationHT() const override { return l1track3D_->cellLocationHT(); }

    //--- Get information about its association (if any) to a truth Tracking Particle.
    //--- Can differ from that of corresponding HT track, if track fit kicked out stubs with bad residuals.

    // Get best matching tracking particle (=nullptr if none).
    const TP* matchedTP() const override { return matchedTP_; }
    // Get the matched stubs with this Tracking Particle
    const std::vector<const Stub*>& matchedStubs() const override { return matchedStubs_; }
    // Get number of matched stubs with this Tracking Particle
    unsigned int numMatchedStubs() const override { return matchedStubs_.size(); }
    // Get number of tracker layers with matched stubs with this Tracking Particle
    unsigned int numMatchedLayers() const override { return nMatchedLayers_; }
    // Get purity of stubs on track (i.e. fraction matching best Tracking Particle)
    float purity() const { return numMatchedStubs() / float(numStubs()); }
    // Get number of stubs matched to correct TP that were deleted from track candidate by fitter.
    unsigned int numKilledMatchedStubs() const {
      unsigned int nStubCount = l1track3D_->numMatchedStubs();
      if (nStubCount > 0) {  // Original HT track candidate did match a truth particle
        const TP* tp = l1track3D_->matchedTP();
        for (const Stub* s : stubs_) {
          std::set<const TP*> assTPs = s->assocTPs();
          if (assTPs.find(tp) != assTPs.end())
            nStubCount--;  // We found a stub matched to original truth particle that survived fit.
        }
      }
      return nStubCount;
    }

    //--- Get the fitted track helix parameters.

    float qOverPt() const override { return qOverPt_; }
    float charge() const { return (qOverPt_ > 0 ? 1 : -1); }
    float invPt() const { return std::abs(qOverPt_); }
    // Protect pt against 1/pt = 0.
    float pt() const {
      constexpr float small = 1.0e-6;
      return 1. / (small + this->invPt());
    }
    float d0() const { return d0_; }
    float phi0() const override { return phi0_; }
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
    // Protect pt against 1/pt = 0.
    float pt_bcon() const {
      constexpr float small = 1.0e-6;
      return 1. / (small + this->invPt_bcon());
    }
    float phi0_bcon() const { return phi0_bcon_; }
    float d0_bcon() const { return d0_bcon_; }

    // Phi and z coordinates at which track crosses "chosenR" values used by r-phi HT and rapidity sectors respectively.
    // (Optionally with beam-spot constraint applied).
    float phiAtChosenR(bool beamConstraint) const {
      if (beamConstraint) {
        return reco::deltaPhi(phi0_bcon_ - ((settings_->invPtToDphi() * settings_->chosenRofPhi()) * qOverPt_bcon_) -
                                  d0_bcon_ / (settings_->chosenRofPhi()),
                              0.);
      } else {
        return reco::deltaPhi(phi0_ - ((settings_->invPtToDphi() * settings_->chosenRofPhi()) * qOverPt_) -
                                  d0_ / (settings_->chosenRofPhi()),
                              0.);
      }
    }
    float zAtChosenR() const {
      return (z0_ + (settings_->chosenRofZ()) * tanLambda_);
    }  // neglects transverse impact parameter & track curvature.

    // Get the number of helix parameters being fitted (=5 if d0 is fitted or =4 if d0 is not fitted).
    float nHelixParam() const { return nHelixParam_; }

    // Get the fit degrees of freedom, chi2 & chi2/DOF (also in r-phi & r-z planes).
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
    unsigned int iPhiSec() const override { return iPhiSec_; }
    unsigned int iEtaReg() const override { return iEtaReg_; }

    //--- Opto-link ID used to send this track from HT to Track Fitter
    unsigned int optoLinkID() const override { return optoLinkID_; }

    //--- Get whether the track has been rejected or accepted by the fit

    bool accepted() const { return accepted_; }

    //--- Functions to help eliminate duplicate tracks.

    // Is the fitted track trajectory should lie within the same HT cell in which the track was originally found?
    bool consistentHTcell() const { return consistentCell_; }

    // Determine if the fitted track trajectory should lie within the same HT cell in which the track was originally found?
    void setConsistentHTcell() {
      // Use helix params with beam-spot constaint if done in case of 5 param fit.

      std::pair<unsigned int, unsigned int> htCell = this->cellLocationHT();
      bool consistent = (htCell == this->cellLocationFit());

      if (l1track3D_->mergedHTcell()) {
        // If this is a merged cell, check other elements of merged cell.
        std::pair<unsigned int, unsigned int> htCell10(htCell.first + 1, htCell.second);
        std::pair<unsigned int, unsigned int> htCell01(htCell.first, htCell.second + 1);
        std::pair<unsigned int, unsigned int> htCell11(htCell.first + 1, htCell.second + 1);
        if (htCell10 == this->cellLocationFit())
          consistent = true;
        if (htCell01 == this->cellLocationFit())
          consistent = true;
        if (htCell11 == this->cellLocationFit())
          consistent = true;
      }

      consistentCell_ = consistent;
    }

    // Is the fitted track trajectory within the same (eta,phi) sector of the HT used to find it?
    bool consistentSector() const {
      if (settings_->hybrid()) {
        float phiCentre = 2. * M_PI * iPhiSec() / settings_->numPhiSectors();
        float sectorHalfWidth = M_PI / settings_->numPhiSectors();
        bool insidePhi = (std::abs(reco::deltaPhi(this->phiAtChosenR(done_bcon_), phiCentre)) < sectorHalfWidth);
        return insidePhi;
      } else {
        bool insidePhi = (std::abs(reco::deltaPhi(this->phiAtChosenR(done_bcon_), secTmp_->phiCentre())) <
                          secTmp_->sectorHalfWidth());
        bool insideEta =
            (this->zAtChosenR() > secTmp_->zAtChosenR_Min() && this->zAtChosenR() < secTmp_->zAtChosenR_Max());
        return (insidePhi && insideEta);
      }
    }

    // Digitize track and degrade helix parameter resolution according to effect of digitisation.
    void digitizeTrack(const std::string& fitterName);

    // Access to detailed info about digitized track. (Gets nullptr if trk not digitized)
    const DigitalTrack* digitaltrack() const { return digitalTrack_.get(); }

  private:
    //--- Configuration parameters
    const Settings* settings_;

    //--- The 3D hough-transform track candidate which was fitted.
    const L1track3D* l1track3D_;

    //--- The stubs on the fitted track (can differ from those on HT track if fit kicked off stubs with bad residuals)
    std::vector<Stub*> stubs_;
    std::vector<const Stub*> stubsConst_;
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

    //--- Sector class used to check if fitted track trajectory is in same sector as HT used to find it.
    std::shared_ptr<Sector> secTmp_;  // shared so as to allow copy of L1fittedTrack.
    //--- r-phi HT class used to determine HT cell location that corresponds to fitted track helix parameters.
    std::shared_ptr<HTrphi> htRphiTmp_;

    //--- Info specific to KF fitter.
    unsigned int nSkippedLayers_;
    unsigned int numUpdateCalls_;
    //--- Info specific to LR fitter.
    int numIterations_;
    std::string lostMatchingState_;
    std::unordered_map<std::string, int> stateCalls_;

    std::shared_ptr<DigitalTrack> digitalTrack_;  // Class used to digitize track if required.

    static const std::vector<Stub*> noStubs_;  // Empty vector used to initialize rejected tracks.

    bool consistentCell_;

    //--- Has the track fit declared this to be a valid track?
    bool accepted_;
  };

}  // namespace tmtt

#endif
