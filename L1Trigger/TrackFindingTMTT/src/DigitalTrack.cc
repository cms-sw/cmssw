#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/DigitalTrack.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <map>

namespace TMTT {

//=== Note configuration parameters.

DigitalTrack::DigitalTrack(const Settings* settings) :

  // Check DigitalTrack correctly initialized.
  ranInit_ (false),
  ranMake_ (false),

  // Digitization configuration parameters
  settings_(settings),

  // Number of phi sectors and phi nonants.
  numPhiSectors_ (settings->numPhiSectors()),
  numPhiNonants_ (settings->numPhiNonants()),
  // Phi sector and phi nonant width (radians)
  phiSectorWidth_(2.*M_PI / float(numPhiSectors_)), 
  phiNonantWidth_(2.*M_PI / float(numPhiNonants_)), 
  // Radius from beamline with respect to which stub r coord. is measured.
  chosenRofPhi_  (settings->chosenRofPhi()),

  // Number of q/Pt bins in Hough  transform array.
  nbinsPt_       ((int) settings->houghNbinsPt()),
  invPtToDPhi_   (settings->invPtToDphi())
{
}

//=== Get digitisation configuration parameters for the specific track fitter being used here.

void DigitalTrack::getDigiCfg(const string& fitterName) {
  if (fitterName == "SimpleLR") {
    // SimpleLR track fitter
    skipTrackDigi_  = settings_->slr_skipTrackDigi();
    oneOver2rBits_  = settings_->slr_oneOver2rBits();
    oneOver2rRange_ = settings_->slr_oneOver2rRange();
    d0Bits_         = settings_->slr_d0Bits();
    d0Range_        = settings_->slr_d0Range();
    phi0Bits_       = settings_->slr_phi0Bits();
    phi0Range_      = settings_->slr_phi0Range();
    z0Bits_         = settings_->slr_z0Bits();
    z0Range_        = settings_->slr_z0Range();
    tanLambdaBits_  = settings_->slr_tanlambdaBits();
    tanLambdaRange_ = settings_->slr_tanlambdaRange();
    chisquaredBits_ = settings_->slr_chisquaredBits();
    chisquaredRange_= settings_->slr_chisquaredRange();
  } else {
    // KF track fitter
    // Also used for all other fitters, though unlikely to be correct them them ...
    if (fitterName == "KF4ParamsComb" || fitterName == "KF5ParamsComb" || fitterName == "KF4ParamsCombHLS") {
      skipTrackDigi_  = settings_->kf_skipTrackDigi();
    } else {
      skipTrackDigi_  = settings_->other_skipTrackDigi(); // Allows to skip digitisation for other fitters
    }
    oneOver2rBits_  = settings_->kf_oneOver2rBits();
    oneOver2rRange_ = settings_->kf_oneOver2rRange();
    d0Bits_         = settings_->kf_d0Bits();
    d0Range_        = settings_->kf_d0Range();
    phi0Bits_       = settings_->kf_phi0Bits();
    phi0Range_      = settings_->kf_phi0Range();
    z0Bits_         = settings_->kf_z0Bits();
    z0Range_        = settings_->kf_z0Range();
    tanLambdaBits_  = settings_->kf_tanlambdaBits();
    tanLambdaRange_ = settings_->kf_tanlambdaRange();
    chisquaredBits_ = settings_->kf_chisquaredBits();
    chisquaredRange_= settings_->kf_chisquaredRange();
  }

  // Calculate multipliers to digitize the floating point numbers.
  oneOver2rMult_ = pow(2.,oneOver2rBits_)/oneOver2rRange_;
  d0Mult_        = pow(2.,d0Bits_)/d0Range_;
  phi0Mult_      = pow(2.,phi0Bits_)/phi0Range_;
  z0Mult_        = pow(2.,z0Bits_)/z0Range_;
  tanLambdaMult_ = pow(2.,tanLambdaBits_)/tanLambdaRange_;
  chisquaredMult_= pow(2.,chisquaredBits_)/chisquaredRange_;
}

//=== Initialize track with original, floating point track params

void DigitalTrack::init(const string& fitterName, unsigned int nHelixParams,
			unsigned int iPhiSec, unsigned int iEtaReg, int mbin, int cbin, int mBinhelix, int cBinhelix, 
		        unsigned int hitPattern,
			float qOverPt_orig, float d0_orig, float phi0_orig, float tanLambda_orig, float z0_orig, float chisquaredRphi_orig, float chisquaredRz_orig,
			float qOverPt_bcon_orig, float phi0_bcon_orig, float chisquaredRphi_bcon_orig, // beam-spot constrained values. 
			unsigned int nLayers, bool consistent, bool consistentSect, bool accepted, 
			float tp_qOverPt, float tp_d0, float tp_phi0, float tp_tanLambda, float tp_z0, float tp_eta, 
			int tp_index, bool tp_useForAlgEff, bool tp_useForEff, int tp_pdgId) 
{
  ranInit_ = true; // Note we ran init().

  fitterName_     = fitterName;
  nHelixParams_   = nHelixParams;

  // Get digitisation parameters for this particular track fitter.
  this->getDigiCfg(fitterName);

  double phiCentreSec0 = -M_PI/float(numPhiNonants_) + M_PI/float(numPhiSectors_);
  phiSectorCentre_ = phiSectorWidth_ * float(iPhiSec) + phiCentreSec0; 

  // FIX: Remove this BODGE once BCHI increased to 11 in KFstate.h
  if (chisquaredRphi_orig >= chisquaredRange_) chisquaredRphi_orig = chisquaredRange_ - 0.1;
  if (chisquaredRphi_bcon_orig >= chisquaredRange_) chisquaredRphi_bcon_orig = chisquaredRange_ - 0.1;

  qOverPt_orig_   = qOverPt_orig;
  oneOver2r_orig_ = qOverPt_orig*invPtToDPhi_;
  d0_orig_        = d0_orig;
  phi0_orig_      = phi0_orig;
  phi0rel_orig_   = reco::deltaPhi(phi0_orig_, phiSectorCentre_);
  tanLambda_orig_ = tanLambda_orig;
  z0_orig_        = z0_orig;
  chisquaredRphi_orig_ = chisquaredRphi_orig;
  chisquaredRz_orig_   = chisquaredRz_orig;

  // Same again with beam-spot constraint.
  qOverPt_bcon_orig_   = qOverPt_bcon_orig;
  oneOver2r_bcon_orig_ = qOverPt_bcon_orig*invPtToDPhi_;
  phi0_bcon_orig_      = phi0_bcon_orig;
  phi0rel_bcon_orig_   = reco::deltaPhi(phi0_bcon_orig_, phiSectorCentre_);
  chisquaredRphi_bcon_orig_= chisquaredRphi_bcon_orig;

  nlayers_         = nLayers;
  iPhiSec_         = iPhiSec;
  iEtaReg_         = iEtaReg;
  mBin_            = mbin;
  cBin_            = cbin;
  mBinhelix_       = mBinhelix;
  cBinhelix_       = cBinhelix;
  hitPattern_      = hitPattern;

  consistent_      = consistent;
  consistentSect_  = consistentSect;
  accepted_        = accepted;
  tp_tanLambda_    = tp_tanLambda;
  tp_qoverpt_      = tp_qOverPt;
  tp_pt_           = 1./(1.0e-6 + fabs(tp_qOverPt));
  tp_d0_           = tp_d0;
  tp_eta_          = tp_eta;
  tp_phi0_         = tp_phi0;
  tp_z0_           = tp_z0;
  tp_index_        = tp_index;
  tp_useForAlgEff_ = tp_useForAlgEff;
  tp_useForEff_    = tp_useForEff;
  tp_pdgId_        = tp_pdgId;
}

//=== Digitize track

void DigitalTrack::makeDigitalTrack() {

  if (! ranInit_) throw cms::Exception("DigitalTrack: You forgot to call init() before makeDigitalTrack()!");

  ranMake_ = true; // Note we ran makeDigitalTrack()

  if (skipTrackDigi_) {
    // Optionally skip track digitisaton if done internally inside track fitting code, so
    // retain original helix params.
    iDigi_oneOver2r_  = 0;
    iDigi_d0_         = 0;
    iDigi_phi0rel_    = 0;
    iDigi_tanLambda_  = 0;
    iDigi_z0_         = 0;
    iDigi_chisquaredRphi_ = 0;
    iDigi_chisquaredRz_ = 0;

    iDigi_oneOver2r_bcon_  = 0;
    iDigi_phi0rel_bcon_    = 0;
    iDigi_chisquaredRphi_bcon_ = 0;

    oneOver2r_    = oneOver2r_orig_; 
    qOverPt_      = qOverPt_orig_;
    d0_           = d0_orig_;
    phi0rel_      = phi0rel_orig_;
    phi0_         = phi0_orig_;
    tanLambda_    = tanLambda_orig_;
    z0_           = z0_orig_;
    chisquaredRphi_ = chisquaredRphi_orig_;
    chisquaredRz_   = chisquaredRz_orig_;

    // Same again with beam-spot constraint.
    oneOver2r_bcon_    = oneOver2r_bcon_orig_; 
    qOverPt_bcon_      = qOverPt_bcon_orig_;
    phi0rel_bcon_      = phi0rel_bcon_orig_;
    phi0_bcon_         = phi0_bcon_orig_;
    chisquaredRphi_bcon_   = chisquaredRphi_bcon_orig_;

  } else {

    //--- Digitize variables

    iDigi_oneOver2r_  = floor(oneOver2r_orig_*oneOver2rMult_);
    iDigi_d0_         = floor(d0_orig_*d0Mult_);
    iDigi_phi0rel_    = floor(phi0rel_orig_*phi0Mult_);
    iDigi_tanLambda_  = floor(tanLambda_orig_*tanLambdaMult_);
    iDigi_z0_         = floor(z0_orig_*z0Mult_);
    iDigi_chisquaredRphi_ = floor(chisquaredRphi_orig_*chisquaredMult_);
    iDigi_chisquaredRz_ = floor(chisquaredRz_orig_*chisquaredMult_);

    // If fitted declared track invalid, it will have set its chi2 to very large number. 
    // So truncate it at maximum allowed by digitisation range.
    if ( ! accepted_ ) {
      iDigi_chisquaredRphi_ = pow(2.,chisquaredBits_) - 1;
      iDigi_chisquaredRz_   = pow(2.,chisquaredBits_) - 1;
    }

    // Same again with beam-spot constraint.
    iDigi_oneOver2r_bcon_  = floor(oneOver2r_bcon_orig_*oneOver2rMult_);
    iDigi_phi0rel_bcon_    = floor(phi0rel_bcon_orig_*phi0Mult_);
    iDigi_chisquaredRphi_bcon_ = floor(chisquaredRphi_bcon_orig_*chisquaredMult_);
    if ( ! accepted_ ) iDigi_chisquaredRphi_bcon_ = pow(2.,chisquaredBits_) - 1;

    // if(settings_->digitizeSLR()){
    //   mBinhelix_ = floor(iDigi_1over2r_/pow(2,5));
    //   cBinhelix_ = floor(iDigi_phiT_/pow(2,7));
    // }

    //--- Determine floating point track params from digitized numbers (so with degraded resolution).

    oneOver2r_    = (iDigi_oneOver2r_ + 0.5)/oneOver2rMult_;
    qOverPt_      = oneOver2r_/invPtToDPhi_;
    if (nHelixParams_ == 5) {
      d0_         = (iDigi_d0_ + 0.5)/d0Mult_;
    } else {
      d0_         = 0.;
    }
    phi0rel_      = (iDigi_phi0rel_ + 0.5)/phi0Mult_;
    phi0_         = reco::deltaPhi(phi0rel_, -phiSectorCentre_);
    tanLambda_    = (iDigi_tanLambda_ + 0.5)/tanLambdaMult_;
    z0_           = (iDigi_z0_ + 0.5)/z0Mult_;
    chisquaredRphi_ = (iDigi_chisquaredRphi_ + 0.5)/chisquaredMult_;
    chisquaredRz_   = (iDigi_chisquaredRz_ + 0.5)/chisquaredMult_;

    // Same again with beam-spot constraint.
    if (nHelixParams_ == 5) {
      oneOver2r_bcon_    = (iDigi_oneOver2r_bcon_ + 0.5)/oneOver2rMult_;
      qOverPt_bcon_      = oneOver2r_bcon_/invPtToDPhi_;
      phi0rel_bcon_      = (iDigi_phi0rel_bcon_ + 0.5)/phi0Mult_;
      phi0_bcon_         = reco::deltaPhi(phi0rel_bcon_, -phiSectorCentre_);
      chisquaredRphi_bcon_   = (iDigi_chisquaredRphi_bcon_ + 0.5)/chisquaredMult_;
    } else {
      oneOver2r_bcon_    = oneOver2r_;
      qOverPt_bcon_      = qOverPt_;
      phi0rel_bcon_      = phi0rel_;
      phi0_bcon_         = phi0_;
      chisquaredRphi_bcon_   = chisquaredRphi_;
    }

    // Check that track coords. are within assumed digitization range.
    this->checkInRange();

    // Check that digitization followed by undigitization doesn't change results too much.
    this->checkAccuracy();
  }
}

//=== Check that stub coords. are within assumed digitization range.

void DigitalTrack::checkInRange() const {
  if (accepted_) { // Don't bother apply to tracks rejected by the fitter.
    if (fabs(oneOver2r_orig_) >= 0.5*oneOver2rRange_)   throw cms::Exception("DigitalTrack: Track oneOver2r is out of assumed digitization range.")<<" |oneOver2r| = " <<fabs(oneOver2r_orig_) <<" > "<<0.5*oneOver2rRange_<<"; Fitter="<<fitterName_<<"; track accepted = "<<accepted_<<endl;  
    if (consistentSect_) { // don't bother if track will fail sector consistency cut.
      if (fabs(phi0rel_orig_) >= 0.5*phi0Range_)   throw cms::Exception("DigitalTrack: Track phi0rel is out of assumed digitization range.")<<" |phi0rel| = " <<fabs(phi0rel_orig_) <<" > "<<0.5*phi0Range_<<"; Fitter="<<fitterName_<<"; track accepted = "<<accepted_<<endl; 
    } 
    if (fabs(z0_orig_) >= 0.5*z0Range_)   throw cms::Exception("DigitalTrack:  Track z0 is out of assumed digitization range.")<<" |z0| = " <<fabs(z0_orig_) <<" > "<<0.5*z0Range_<<"; Fitter="<<fitterName_<<"; track accepted = "<<accepted_<<endl;  
    if (fabs(d0_orig_) >= 0.5*d0Range_)   throw cms::Exception("DigitalTrack:  Track d0 is out of assumed digitization range.")<<" |d0| = " <<fabs(d0_orig_) <<" > "<<0.5*d0Range_<<"; Fitter="<<fitterName_<<"; track accepted = "<<accepted_<<endl;  
    if (fabs(tanLambda_orig_) >= 0.5*tanLambdaRange_)   throw cms::Exception("DigitalTrack: Track tanLambda is out of assumed digitization range.")<<" |tanLambda| = " <<fabs(tanLambda_orig_) <<" > "<<0.5*tanLambdaRange_<<"; Fitter="<<fitterName_<<"; track accepted = "<<accepted_<<endl;  
    if (accepted_) { // Tracks declared invalid by fitter can have very large original chi2.
      if (chisquaredRphi_orig_ >= chisquaredRange_ or chisquaredRphi_orig_ < 0.)   throw cms::Exception("DigitalTrack: Track chisquaredRphi is out of assumed digitization range.")<<" chisquaredRphi = " <<chisquaredRphi_orig_ <<" > "<<chisquaredRange_<<" or < 0"<<"; Fitter="<<fitterName_<<"; track accepted = "<<accepted_<<endl;  
      if (chisquaredRz_orig_ >= chisquaredRange_ or chisquaredRz_orig_ < 0.)   throw cms::Exception("DigitalTrack: Track chisquaredRz is out of assumed digitization range.")<<" chisquaredRz = " <<chisquaredRz_orig_ <<" > "<<chisquaredRange_<<" or < 0"<<"; Fitter="<<fitterName_<<"; track accepted = "<<accepted_<<endl;  
    }
  }
}

//=== Check that digitisation followed by undigitisation doesn't change significantly the stub coordinates.

void DigitalTrack::checkAccuracy() const {
  if (accepted_) { // Don't bother apply to tracks rejected by the fitter.
    float TA = qOverPt_- qOverPt_orig_;
    float TB = reco::deltaPhi(phi0_, phi0_orig_);
    float TC = z0_ - z0_orig_;
    float TD = tanLambda_ - tanLambda_orig_;
    float TE = d0_ - d0_orig_;
    float TF = chisquaredRphi_  - chisquaredRphi_orig_;
    float TG = chisquaredRz_    - chisquaredRz_orig_;

    static map<string, unsigned int> nErr; // Count precision errors from each fitter.
    if (nErr.find(fitterName_) == nErr.end()) nErr[fitterName_] = 0; // Initialize error count.
    const  unsigned int maxErr = 20;  // Print error message only this number of times.
    if (nErr[fitterName_] < maxErr) {
      if (fabs(TA) > 0.01 || fabs(TB) > 0.001 || fabs(TC) > 0.05 || fabs(TD) > 0.002 || fabs(TE) > 0.05 || fabs(TF) > 0.5 || fabs(TG) > 0.5) {
	nErr[fitterName_]++;
	cout<<"WARNING: DigitalTrack lost precision: "<<fitterName_<<" accepted="<<accepted_<<" "<<TA<<" "<<TB<<" "<<TC<<" "<<TD<<" "<<TE<<" "<<TF<<" "<<TG<<endl;
      }
    }
  }
}

}
