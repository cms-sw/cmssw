#ifndef L1Trigger_TrackFindingTMTT_DigitalStub_h
#define L1Trigger_TrackFindingTMTT_DigitalStub_h

#include "FWCore/Utilities/interface/Exception.h"
#include <cmath>
#include <string>

//=== Digtizes stubs for input to GP, HT & KF

namespace tmtt {

  class Settings;

  class DigitalStub {
  public:
    //--- Hybrid tracking: simplified digitization for KF.

    DigitalStub(const Settings* settings, double r, double phi, double z, unsigned int iPhiSec);

    //--- TMTT tracking:
    // Initialize stub with floating point stub coords, range of HT m-bin values consistent with bend,
    // bend and phi sector.
    DigitalStub(const Settings* settings,
                double phi_orig,
                double r_orig,
                double z_orig,
                unsigned int mbin_min_orig,
                unsigned int mbin_max_orig,
                double bend_orig,
                unsigned int iPhiSec);

    // Redo phi digitisation assigning stub to a different phi sector;
    // (Return arg indicates if any work done).
    bool changePhiSec(unsigned int iPhiSec);

    //--- Original floating point stub data before digitization.
    double r_orig() const { return r_orig_; }
    double rt_orig() const { return rt_orig_; }  // r with respect to reference radius
    double phi_orig() const { return phi_orig_; }
    double phiS_orig() const { return phiS_orig_; }  // with respect to centre of sector
    double phiN_orig() const { return phiN_orig_; }  // with respect to centre of nonant
    double z_orig() const { return z_orig_; }
    unsigned int mbin_min_orig() const { return mbin_min_orig_; }
    unsigned int mbin_max_orig() const { return mbin_max_orig_; }
    double bend_orig() const { return bend_orig_; }

    //--- Digitised stub data

    int iDigi_PhiN() const { return iDigi_PhiN_; }
    int iDigi_Bend() const { return iDigi_Bend_; }
    int iDigi_PhiS() const { return iDigi_PhiS_; }
    int mbin_min() const { return mbin_min_; }
    int mbin_max() const { return mbin_max_; }
    int iDigi_Rt() const { return iDigi_Rt_; }
    unsigned int iDigi_R() const { return iDigi_R_; }
    int iDigi_Z() const { return iDigi_Z_; }

    //--- floating point stub data following digitisation & undigitisation again
    // "GP" indicates valid variable for input to GP etc. If no such name, always valid.
    double phiN() const { return phiN_; }
    double phi_GP() const { return phi_GP_; }
    double bend() const { return bend_; }
    double phiS() const { return phiS_; }
    double phi_HT_TF() const { return phi_HT_TF_; }
    double rt_GP_HT() const { return rt_GP_HT_; }
    double r_GP_HT() const { return r_GP_HT_; }
    double r_SF_TF() const { return r_SF_TF_; }
    double rt_SF_TF() const { return rt_SF_TF_; }
    double z() const { return z_; }

    //--- Utility: return phi nonant number corresponding to given phi sector number.

    unsigned int iNonant(unsigned int iPhiSec) const { return floor(iPhiSec * numPhiNonants_ / numPhiSectors_); }

  private:
    // Set cfg params.
    void setCfgParams(const Settings* settings);

    // Digitize stub
    void digitize(unsigned int iPhiSec);

    // Undigitize stub again
    void undigitize(unsigned int iPhiSec);

    // Check that stub coords. & bend angle are within assumed digitization range.
    void checkInRange() const;

    // Check that digitisation followed by undigitisation doesn't change significantly the stub coordinates.
    void checkAccuracy() const;

  private:
    //--- Configuration

    // Digitization configuration
    int iFirmwareType_;
    unsigned int phiSectorBits_;
    unsigned int phiSBits_;
    double phiSRange_;
    unsigned int rtBits_;
    double rtRange_;
    unsigned int zBits_;
    double zRange_;
    unsigned int phiNBits_;
    double phiNRange_;
    unsigned int bendBits_;
    double bendRange_;

    // Digitization multipliers
    double phiSMult_;
    double rtMult_;
    double zMult_;
    double phiNMult_;
    double bendMult_;

    // Number of phi sectors and phi nonants.
    unsigned int numPhiSectors_;
    unsigned int numPhiNonants_;
    // Phi sector and phi nonant width (radians)
    double phiSectorWidth_;
    double phiNonantWidth_;
    // Centre of phi sector 0.
    double phiCentreSec0_;
    // Centre of this phi sector & nonant.
    double phiSectorCentre_;
    double phiNonantCentre_;
    // Radius from beamline with respect to which stub r coord. is measured.
    double chosenRofPhi_;
    // Number of q/Pt bins in Hough  transform array.
    unsigned int nbinsPt_;
    // Min. of HT m-bin array in firmware.
    int min_array_mbin_;

    // Used to check if new digitisation requests were already done.
    unsigned int iPhiSec_done_;

    //--- Original floating point stub data before digitization.
    double r_orig_;
    double rt_orig_;
    double phi_orig_;
    double phiS_orig_;
    double phiN_orig_;
    double z_orig_;
    unsigned int mbin_min_orig_;
    unsigned int mbin_max_orig_;
    double bend_orig_;

    //--- Digitised stub data

    int iDigi_PhiN_;
    int iDigi_Bend_;
    int iDigi_PhiS_;
    int mbin_min_;
    int mbin_max_;
    int iDigi_Rt_;
    unsigned int iDigi_R_;
    int iDigi_Z_;

    //--- floating point stub data following digitisation & undigitisation again
    double phiN_;
    double phi_GP_;
    double bend_;
    double phiS_;
    double phi_HT_TF_;
    double rt_GP_HT_;
    double r_GP_HT_;
    double r_SF_TF_;
    double rt_SF_TF_;
    double z_;
  };

}  // namespace tmtt
#endif
