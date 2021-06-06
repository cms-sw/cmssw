#ifndef L1Trigger_TrackFindingTMTT_DigitalTrack_h
#define L1Trigger_TrackFindingTMTT_DigitalTrack_h

#include "FWCore/Utilities/interface/Exception.h"
#include <cmath>
#include <string>
#include <set>

namespace tmtt {

  class Settings;
  class L1fittedTrack;

  //====================================================================================================
  /**
 * Used to digitize the fitted track helix params.
 * WARNING: Digitizes according to common format agreed for KF and SimpleLR4 fitters, 
 * and uses KF digitisation cfg for all fitters except SimpleLR4.
 */
  //====================================================================================================

  class DigitalTrack {
  public:
    // Digitize track
    DigitalTrack(const Settings* settings, const std::string& fitterName, const L1fittedTrack* fitTrk);

    //--- The functions below return variables post-digitization.

    // half inverse curvature of track.
    int iDigi_oneOver2r() const { return iDigi_oneOver2r_; }
    int iDigi_d0() const { return iDigi_d0_; }
    // measured relative to centre of sector
    int iDigi_phi0rel() const { return iDigi_phi0rel_; }
    int iDigi_z0() const { return iDigi_z0_; }
    int iDigi_tanLambda() const { return iDigi_tanLambda_; }
    unsigned int iDigi_chisquaredRphi() const { return iDigi_chisquaredRphi_; }
    unsigned int iDigi_chisquaredRz() const { return iDigi_chisquaredRz_; }

    // Digits corresponding to track params with post-fit beam-spot constraint.
    int iDigi_oneOver2r_bcon() const { return iDigi_oneOver2r_bcon_; }  // half inverse curvature of track.
    int iDigi_phi0rel_bcon() const { return iDigi_phi0rel_bcon_; }      // measured relative to centre of sector
    unsigned int iDigi_chisquaredRphi_bcon() const { return iDigi_chisquaredRphi_bcon_; }

    // Floating point track params derived from digitized info (so with degraded resolution).
    float qOverPt() const { return qOverPt_; }
    float oneOver2r() const { return oneOver2r_; }  // half inverse curvature of track.
    float d0() const { return d0_; }
    float phi0() const { return phi0_; }
    float phi0rel() const { return phi0rel_; }  // measured relative to centre of sector
    float z0() const { return z0_; }
    float tanLambda() const { return tanLambda_; }
    float chisquaredRphi() const { return chisquaredRphi_; }
    float chisquaredRz() const { return chisquaredRz_; }

    // Floating point track params derived from digitized track params with post-fit beam-spot constraint.
    float qOverPt_bcon() const { return qOverPt_bcon_; }
    float oneOver2r_bcon() const { return oneOver2r_bcon_; }  // half inverse curvature of track.
    float phi0_bcon() const { return phi0_bcon_; }
    float phi0rel_bcon() const { return phi0rel_bcon_; }  // measured relative to centre of sector
    float chisquaredRphi_bcon() const { return chisquaredRphi_bcon_; }

    unsigned int iPhiSec() const { return iPhiSec_; }
    unsigned int iEtaReg() const { return iEtaReg_; }
    int mBinhelix() const { return mBinhelix_; }
    int cBinhelix() const { return cBinhelix_; }
    unsigned int nlayers() const { return nlayers_; }
    int mBinHT() const { return mBin_; }
    int cBinHT() const { return cBin_; }
    bool accepted() const { return accepted_; }
    unsigned int hitPattern() const { return hitPattern_; }

    //--- The functions below give access to the original variables prior to digitization.
    //%%% Those common to GP & HT input.
    float orig_qOverPt() const { return qOverPt_orig_; }
    float orig_oneOver2r() const { return oneOver2r_orig_; }  // half inverse curvature of track.
    float orig_d0() const { return d0_orig_; }
    float orig_phi0() const { return phi0_orig_; }
    float orig_phi0rel() const { return phi0rel_orig_; }  // measured relative to centre of sector
    float orig_z0() const { return z0_orig_; }
    float orig_tanLambda() const { return tanLambda_orig_; }
    float orig_chisquaredRphi() const { return chisquaredRphi_orig_; }
    float orig_chisquaredRz() const { return chisquaredRz_orig_; }

    float tp_pt() const { return tp_pt_; }
    float tp_eta() const { return tp_eta_; }
    float tp_d0() const { return tp_d0_; }
    float tp_phi0() const { return tp_phi0_; }
    float tp_tanLambda() const { return tp_tanLambda_; }
    float tp_z0() const { return tp_z0_; }
    float tp_qoverpt() const { return tp_qoverpt_; }
    int tp_index() const { return tp_index_; }
    float tp_useForAlgEff() const { return tp_useForAlgEff_; }
    float tp_useForEff() const { return tp_useForEff_; }
    float tp_pdgId() const { return tp_pdgId_; }

    //--- Utility: return phi nonant number corresponding to given phi sector number.
    unsigned int iGetNonant(unsigned int iPhiSec) const { return floor(iPhiSec * numPhiNonants_ / numPhiSectors_); }

  private:
    // Load digitisation configuration parameters for the specific track fitter being used here.
    void loadDigiCfg(const std::string& fitterName);

    // Digitize track
    void makeDigitalTrack();

    // Check that stub coords. are within assumed digitization range.
    void checkInRange() const;

    // Check that digitisation followed by undigitisation doesn't change significantly the track params.
    void checkAccuracy() const;

  private:
    // Configuration params
    const Settings* settings_;

    // Number of phi sectors and phi nonants.
    unsigned int numPhiSectors_;
    unsigned int numPhiNonants_;
    double phiSectorWidth_;
    double phiNonantWidth_;
    double phiSectorCentre_;
    float chosenRofPhi_;
    unsigned int nbinsPt_;
    float invPtToDPhi_;

    // Digitization configuration
    bool skipTrackDigi_;
    unsigned int oneOver2rBits_;
    float oneOver2rRange_;
    unsigned int d0Bits_;
    float d0Range_;
    unsigned int phi0Bits_;
    float phi0Range_;
    unsigned int z0Bits_;
    float z0Range_;
    unsigned int tanLambdaBits_;
    float tanLambdaRange_;
    unsigned int chisquaredBits_;
    float chisquaredRange_;

    double oneOver2rMult_;
    double d0Mult_;
    double phi0Mult_;
    double z0Mult_;
    double tanLambdaMult_;
    double chisquaredMult_;

    // Fitted track

    std::string fitterName_;
    unsigned int nHelixParams_;

    // Integer data after digitization (which doesn't degrade its resolution, but can recast it in a different form).
    unsigned int nlayers_;
    unsigned int iPhiSec_;
    unsigned int iEtaReg_;
    int mBin_;
    int cBin_;
    int mBinhelix_;
    int cBinhelix_;
    unsigned int hitPattern_;
    bool consistent_;
    bool consistentSect_;
    bool accepted_;

    //--- Original floating point stub coords before digitization.

    float qOverPt_orig_;
    float oneOver2r_orig_;
    float d0_orig_;
    float phi0_orig_;
    float phi0rel_orig_;
    float tanLambda_orig_;
    float z0_orig_;
    float chisquaredRphi_orig_;
    float chisquaredRz_orig_;

    float qOverPt_bcon_orig_;
    float oneOver2r_bcon_orig_;
    float phi0_bcon_orig_;
    float phi0rel_bcon_orig_;
    float chisquaredRphi_bcon_orig_;

    //--- Digits corresponding to track params.

    int iDigi_oneOver2r_;
    int iDigi_d0_;
    int iDigi_phi0rel_;
    int iDigi_z0_;
    int iDigi_tanLambda_;
    unsigned int iDigi_chisquaredRphi_;
    unsigned int iDigi_chisquaredRz_;

    int iDigi_oneOver2r_bcon_;
    int iDigi_phi0rel_bcon_;
    unsigned int iDigi_chisquaredRphi_bcon_;

    //--- Floating point track coords derived from digitized info (so with degraded resolution).

    float qOverPt_;
    float oneOver2r_;
    float d0_;
    float phi0_;
    float phi0rel_;
    float z0_;
    float tanLambda_;
    float chisquaredRphi_;
    float chisquaredRz_;

    float qOverPt_bcon_;
    float oneOver2r_bcon_;
    float phi0_bcon_;
    float phi0rel_bcon_;
    float chisquaredRphi_bcon_;

    // Truth
    float tp_qoverpt_;
    float tp_pt_;
    float tp_eta_;
    float tp_d0_;
    float tp_phi0_;
    float tp_tanLambda_;
    float tp_z0_;
    int tp_index_;
    bool tp_useForAlgEff_;
    bool tp_useForEff_;
    int tp_pdgId_;
  };

}  // namespace tmtt
#endif
