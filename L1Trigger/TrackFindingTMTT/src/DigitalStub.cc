#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/DigitalStub.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <atomic>

using namespace std;

namespace tmtt {

  //=== Hybrid tracking: simplified digitization for KF.

  DigitalStub::DigitalStub(const Settings* settings, double r, double phi, double z, unsigned int iPhiSec)
      : phiSBits_(settings->phiSBits()),    // No. of bits to store phiS coord.
        phiSRange_(settings->phiSRange()),  // Range of phiS coord. in radians.
        rtBits_(settings->rtBits()),        // No. of bits to store rT coord.
        rtRange_(settings->rtRange()),      // Range of rT coord. in cm.
        zBits_(settings->zBits()),          // No. of bits to store z coord.
        zRange_(settings->zRange()),        // Range of z coord in cm.
        phiSMult_(pow(2, phiSBits_) / phiSRange_),
        rtMult_(pow(2, rtBits_) / rtRange_),
        zMult_(pow(2, zBits_) / zRange_),
        numPhiSectors_(settings->numPhiSectors()),
        numPhiNonants_(9),
        phiSectorWidth_(2. * M_PI / double(numPhiSectors_)),
        chosenRofPhi_(settings->chosenRofPhi()) {
    // Centre of this sector in phi. (Nonant 0 is centred on x-axis).
    phiCentreSec0_ = -M_PI / double(numPhiNonants_) + M_PI / double(numPhiSectors_);
    phiSectorCentre_ = phiSectorWidth_ * double(iPhiSec) + phiCentreSec0_;

    // Used to check if new digi requests are for same sector as old
    iPhiSec_done_ = iPhiSec;

    r_orig_ = r;
    phi_orig_ = phi;
    z_orig_ = z;

    rt_orig_ = r_orig_ - chosenRofPhi_;
    phiS_orig_ = reco::deltaPhi(phi_orig_, phiSectorCentre_);

    // Digitize
    iDigi_Rt_ = floor(rt_orig_ * rtMult_);
    iDigi_PhiS_ = floor(phiS_orig_ * phiSMult_);
    iDigi_Z_ = floor(z_orig_ * zMult_);
  }

  //=== TMTT tracking algorithm: digitisaton for entire L1 tracking chain.
  // Initialize stub with floating point stub coords, range of HT m-bin values consistent with bend,
  // bend and phi sector.

  DigitalStub::DigitalStub(const Settings* settings,
                           double phi_orig,
                           double r_orig,
                           double z_orig,
                           unsigned int mbin_min_orig,
                           unsigned int mbin_max_orig,
                           double bend_orig,
                           unsigned int iPhiSec) {
    // Set cfg params
    this->setCfgParams(settings);

    // Store variable prior to digitisation
    r_orig_ = r_orig;
    rt_orig_ = r_orig_ - chosenRofPhi_;
    phi_orig_ = phi_orig;
    z_orig_ = z_orig;
    mbin_min_orig_ = mbin_min_orig;
    mbin_max_orig_ = mbin_max_orig;
    bend_orig_ = bend_orig;

    // Phi of centre of phi sector and of nonant.
    unsigned int iNonant = this->iNonant(iPhiSec);
    phiSectorCentre_ = phiSectorWidth_ * double(iPhiSec) + phiCentreSec0_;
    phiNonantCentre_ = phiNonantWidth_ * double(iNonant);

    phiS_orig_ = reco::deltaPhi(phi_orig_, phiSectorCentre_);
    phiN_orig_ = reco::deltaPhi(phi_orig_, phiNonantCentre_);

    // Used to check if new digi requests are for same sector as old
    iPhiSec_done_ = iPhiSec;

    // Check that stub coords. are within assumed digitization range.
    this->checkInRange();

    // Digitize and then undigitize stub.
    this->digitize(iPhiSec);
    this->undigitize(iPhiSec);

    // Check that digitization followed by undigitization doesn't change results too much.
    this->checkAccuracy();
  }

  //=== Redo phi digitisation assigning stub to a different phi sector;

  bool DigitalStub::changePhiSec(unsigned int iPhiSec) {
    bool doUpdate = (iPhiSec != iPhiSec_done_);

    if (doUpdate) {
      // phi sector has changed since last time digitisation was done, so update.
      iPhiSec_done_ = iPhiSec;
      unsigned int iNonant = this->iNonant(iPhiSec);
      // Update original, floating point phi w.r.t. phi sector/nonant centre.
      phiSectorCentre_ = phiSectorWidth_ * double(iPhiSec) + phiCentreSec0_;
      phiNonantCentre_ = phiNonantWidth_ * double(iNonant);
      phiS_orig_ = reco::deltaPhi(phi_orig_, phiSectorCentre_);
      phiN_orig_ = reco::deltaPhi(phi_orig_, phiNonantCentre_);
      // Update digitised phi.
      iDigi_PhiN_ = floor(phiN_orig_ * phiNMult_);
      iDigi_PhiS_ = floor(phiS_orig_ * phiSMult_);
      // Update digitized then undigitized phi.
      phiN_ = (iDigi_PhiN_ + 0.5) / phiNMult_;
      phi_GP_ = reco::deltaPhi(phiN_, -phiNonantCentre_);
      phiS_ = (iDigi_PhiS_ + 0.5) / phiSMult_;
      phi_HT_TF_ = reco::deltaPhi(phiS_, -phiSectorCentre_);
    }
    return doUpdate;
  }

  //=== Set configuration parameters.

  void DigitalStub::setCfgParams(const Settings* settings) {
    // Digitization configuration parameters
    phiSectorBits_ = settings->phiSectorBits();  // No. of bits to store phi sector number
    //--- Parameters available in HT board.
    phiSBits_ = settings->phiSBits();    // No. of bits to store phiS coord.
    phiSRange_ = settings->phiSRange();  // Range of phiS coord. in radians.
    rtBits_ = settings->rtBits();        // No. of bits to store rT coord.
    rtRange_ = settings->rtRange();      // Range of rT coord. in cm.
    zBits_ = settings->zBits();          // No. of bits to store z coord.
    zRange_ = settings->zRange();        // Range of z coord in cm.
    //--- Parameters available in GP board (excluding any in common with HT specified above).
    phiNBits_ = settings->phiNBits();    // No. of bits to store phiN parameter.
    phiNRange_ = settings->phiNRange();  // Range of phiN parameter
    bendBits_ = settings->bendBits();    // No. of bits to store stub bend.

    // Number of phi sectors and phi nonants.
    numPhiSectors_ = settings->numPhiSectors();
    numPhiNonants_ = settings->numPhiNonants();
    // Phi sector and phi nonant width (radians)
    phiSectorWidth_ = 2. * M_PI / double(numPhiSectors_);
    phiNonantWidth_ = 2. * M_PI / double(numPhiNonants_);
    // Centre of phi sector 0.
    phiCentreSec0_ = -M_PI / double(numPhiNonants_) + M_PI / double(numPhiSectors_);
    // Radius from beamline with respect to which stub r coord. is measured.
    chosenRofPhi_ = settings->chosenRofPhi();

    // Number of q/Pt bins in Hough  transform array.
    nbinsPt_ = (int)settings->houghNbinsPt();
    // Min. of m-bin range in firmware,
    min_array_mbin_ = (nbinsPt_ % 2 == 0) ? -(nbinsPt_ / 2) : -(nbinsPt_ - 1) / 2;

    // Calculate multipliers to digitize the floating point numbers.
    phiSMult_ = pow(2, phiSBits_) / phiSRange_;
    rtMult_ = pow(2, rtBits_) / rtRange_;
    zMult_ = pow(2, zBits_) / zRange_;
    phiNMult_ = pow(2, phiNBits_) / phiNRange_;

    // No precision lost by digitization, since original bend (after encoding) has steps of 0.25 (in units of pitch).
    bendMult_ = 4.;
    bendRange_ = round(pow(2, bendBits_) / bendMult_);  // discrete values, so digitisation different
  }

  //=== Digitize stub

  void DigitalStub::digitize(unsigned int iPhiSec) {
    //--- Digitize variables used exclusively in GP input.
    iDigi_PhiN_ = floor(phiN_orig_ * phiNMult_);
    iDigi_Bend_ = round(bend_orig_ * bendMult_);  // discrete values, so digitisation different

    //--- Digitize variables used exclusively in HT input.
    iDigi_PhiS_ = floor(phiS_orig_ * phiSMult_);

    // Offset m-bin range allowed by bend to correspond to firmware.
    mbin_min_ = mbin_min_orig_ + min_array_mbin_;
    mbin_max_ = mbin_max_orig_ + min_array_mbin_;

    //--- Digitize variables used in both GP & HT input.
    iDigi_Rt_ = floor(rt_orig_ * rtMult_);

    //-- Digitize variables used by SF & TF input
    iDigi_R_ = iDigi_Rt_ + std::round(chosenRofPhi_ * rtMult_);

    //-- Digitize variables used by everything
    iDigi_Z_ = floor(z_orig_ * zMult_);
  }

  //=== Undigitize stub again.

  void DigitalStub::undigitize(unsigned int iPhiSec) {
    //--- Undigitize variables used exclusively in GP.
    phiN_ = (iDigi_PhiN_ + 0.5) / phiNMult_;
    phi_GP_ = reco::deltaPhi(phiN_, -phiNonantCentre_);
    bend_ = iDigi_Bend_ / bendMult_;  // discrete values, so digitisation different

    //--- Undigitize variables used exclusively by HT & SF/TF
    phiS_ = (iDigi_PhiS_ + 0.5) / phiSMult_;
    phi_HT_TF_ = reco::deltaPhi(phiS_, -phiSectorCentre_);

    //--- Undigitize variables used in both GP & HT.
    rt_GP_HT_ = (iDigi_Rt_ + 0.5) / rtMult_;
    r_GP_HT_ = rt_GP_HT_ + chosenRofPhi_;

    //--- Undigitize variables used exclusively by  SF/TF.
    r_SF_TF_ = (iDigi_R_ + 0.5) / rtMult_;
    rt_SF_TF_ = r_SF_TF_ - chosenRofPhi_;

    //--- Undigitize variables used exclusively by everything.
    z_ = (iDigi_Z_ + 0.5) / zMult_;
  }

  //=== Check that stub coords. are within assumed digitization range.

  void DigitalStub::checkInRange() const {
    if (std::abs(rt_orig_) >= 0.5 * rtRange_)
      throw cms::Exception("BadConfig") << "DigitalStub: Stub rT is out of assumed digitization range."
                                        << " |rt| = " << std::abs(rt_orig_) << " > " << 0.5 * rtRange_;
    if (std::abs(z_orig_) >= 0.5 * zRange_)
      throw cms::Exception("BadConfig") << "DigitalStub: Stub z is out of assumed digitization range."
                                        << " |z| = " << std::abs(z_orig_) << " > " << 0.5 * zRange_;
    if (std::abs(bend_orig_) >= 0.5 * bendRange_)
      throw cms::Exception("BadConfig") << "DigitalStub: Stub bend is out of assumed digitization range."
                                        << " |bend| = " << std::abs(bend_orig_) << " > " << 0.5 * bendRange_;
    //--- Can't check phi range, as DigitalStub called for stubs before sector assignment.
    //if (std::abs(phiS_orig_) >= 0.5 * phiSRange_)
    //  throw cms::Exception("BadConfig") << "DigitalStub: Stub phiS is out of assumed digitization range."
    //				      << " |phiS| = " << std::abs(phiS_orig_) << " > " << 0.5 * phiSRange_;
    //if (std::abs(phiN_orig_) >= 0.5 * phiNRange_)
    //  throw cms::Exception("BadConfig") << "DigitalStub: Stub phiN is out of assumed digitization range."
    //			      << " |phiN| = " << std::abs(phiN_orig_) << " > " << 0.5 * phiNRange_;
  }

  //=== Check that digitisation followed by undigitisation doesn't change significantly the stub coordinates.

  void DigitalStub::checkAccuracy() const {
    double TA = reco::deltaPhi(phi_HT_TF_, phi_orig_);
    double TB = r_GP_HT_ - r_orig_;
    double TC = z_ - z_orig_;
    double TD = bend_ - bend_orig_;

    // Compare to small numbers, representing acceptable precision loss.
    constexpr double smallTA = 0.001, smallTB = 0.3, smallTC = 0.25, smallTD = 0.01;
    if (std::abs(TA) > smallTA || std::abs(TB) > smallTB || std::abs(TC) > smallTC || std::abs(TD) > smallTD) {
      throw cms::Exception("LogicError") << "WARNING: DigitalStub lost precision: " << TA << " " << TB << " " << TC
                                         << " " << TD;
    }
  }

}  // namespace tmtt
