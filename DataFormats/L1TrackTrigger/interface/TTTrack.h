/*! \class   TTTrack
 *  \brief   Class to store the L1 Tracks.
 *  \details The template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon (+ update Ian Tomalin)
 */

#ifndef L1_TRACK_TRIGGER_TRACK_FORMAT_H
#define L1_TRACK_TRIGGER_TRACK_FORMAT_H

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/deltaPhi.h"

template <typename T>
class TTTrack : public TTTrack_TrackWord {
public:
  typedef math::ErrorF<5>::type CovMat;
  typedef edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > TTStubRef;
  enum Hpar { INVR, PHI0, TANL, Z0, D0 };

private:
  /// Data members
  std::vector<TTStubRef> theStubRefs;
  GlobalVector theMomentum_;
  GlobalPoint thePOCA_;
  double theRInv_;
  double thePhi_;
  double theTanL_;
  double theD0_;
  double theZ0_;
  unsigned int thePhiSector_;
  unsigned int theEtaSector_;
  double theStubPtConsistency_;
  double theChi2_;
  double theChi2_XY_;
  double theChi2_Z_;
  unsigned int theNumFitPars_;
  unsigned int theHitPattern_;
  double theTrkMVA1_;
  double theTrkMVA2_;
  double theTrkMVA3_;
  int theTrackSeedType_;
  double theBField_;  // needed for unpacking
  CovMat theHelixCovMat_;

  static constexpr double cLight_ = CLHEP::c_light * (CLHEP::second / CLHEP::meter);

public:
  /// Constructors
  TTTrack();

  TTTrack(double Rinv,
          double phi0,
          double tanL,
          double z0,
          double d0,
          double aChi2xyfit,
          double aChi2zfit,
          double trkMVA1,
          double trkMVA2,
          double trkMVA3,
          unsigned int aHitpattern,  // Bit order: Inner-->Outer tracker corresponds to Rightmost-->Leftmost bits
          unsigned int nPar,
          double Bfield,
          unsigned int phiSector = 99,
          unsigned int etaSector = 99,
          double chi2BendRed = -99,
          unsigned int seedType = 99,
          const CovMat& helixCovMat = CovMat());

  /// Track stubs (not available in firmware)
  const std::vector<TTStubRef>& getStubRefs() const { return theStubRefs; }
  void addStubRef(const TTStubRef& aStub) { theStubRefs.push_back(aStub); }
  void setStubRefs(const std::vector<TTStubRef>& aStubs) { theStubRefs = aStubs; }

  /// Track momentum
  const GlobalVector& momentum() const { return theMomentum_; }
  double pt() const { return theMomentum_.perp(); }

  /// Track curvature
  double rInv() const { return theRInv_; }

  /// Track phi0
  double phi() const { return thePhi_; }

  // Track phi with respect to centre line of phi nonant, which for sector 0 is parallel to x-axis.
  double localPhi() const { return TTTrack_TrackWord::localPhi(thePhi_, thePhiSector_); }

  /// Track tanL
  double tanL() const { return theTanL_; }

  /// Track d0
  double d0() const { return theD0_; }

  /// Track z0
  double z0() const { return theZ0_; }

  /// Track eta
  double eta() const { return theMomentum_.eta(); }

  /// Track charge
  int charge() const { return (theRInv_ > 0) ? 1 : -1; }

  // Helix covariance matrix (always 5x5) in (1/R, phi0, tanL, z0, d0),
  // so elements can be accessed with enum Hpar.
  // (Added to study if any elements worth adding to L1 track firmware output).
  const CovMat& helixCovMat() const { return theHelixCovMat_; }

  /// POCA
  const GlobalPoint& POCA() const { return thePOCA_; }

  /// MVA Track quality variables
  double trkMVA1() const { return theTrkMVA1_; }  // Tuned for prompt tracks
  void settrkMVA1(double atrkMVA1) { theTrkMVA1_ = atrkMVA1; }
  double trkMVA2() const { return theTrkMVA2_; }  // Tuned for prompt electron tracks
  void settrkMVA2(double atrkMVA2) { theTrkMVA2_ = atrkMVA2; }
  double trkMVA3() const { return theTrkMVA3_; }  // Tuned for displaced tracks
  void settrkMVA3(double atrkMVA3) { theTrkMVA3_ = atrkMVA3; }

  /// Phi Sector
  unsigned int phiSector() const { return thePhiSector_; }
  // Obsolete. Use constructor instead.
  void setPhiSector(unsigned int aSector) { thePhiSector_ = aSector; }

  /// Eta Sector
  unsigned int etaSector() const { return theEtaSector_; }
  // Obsolete. Use constructor instead.
  void setEtaSector(unsigned int aSector) { theEtaSector_ = aSector; }

  /// Track seeding (for debugging)
  unsigned int trackSeedType() const { return theTrackSeedType_; }
  // Obsolete. Use constructor instead.
  void setTrackSeedType(int aSeed) { theTrackSeedType_ = aSeed; }

  /// Chi2 and chi2/ndf ("Red")
  double chi2() const { return theChi2_; }
  double chi2Red() const { return theChi2_ / (2 * theStubRefs.size() - theNumFitPars_); }

  // Ditto, but split into r-z and x-y components.
  double chi2Z() const { return theChi2_Z_; }
  double chi2ZRed() const {
    constexpr int nHelixParsRZ = 2;
    return theChi2_Z_ / (theStubRefs.size() - nHelixParsRZ);
  }
  double chi2XY() const { return theChi2_XY_; }
  double chi2XYRed() const {
    int nHelixParsRPhi = theNumFitPars_ - 2;
    return theChi2_XY_ / (theStubRefs.size() - nHelixParsRPhi);
  }

  /// Stub Pt consistency (i.e. stub bend chi2/dof)
  double chi2BendRed() const { return theStubPtConsistency_; }
  double chi2Bend() const { return chi2BendRed() * theStubRefs.size(); }
  // Obsolete. Use constructor instead
  void setChi2BendRed(double aChi2BendRed) { theStubPtConsistency_ = aChi2BendRed; }

  int nFitPars() const { return theNumFitPars_; }

  /// Hit Pattern
  unsigned int hitPattern() const { return theHitPattern_; }

  /// Get helix params & chi2XY after constraining track to x=y=0.
  //  N.B. Should really constrain to beam-line position!
  // (Added to study if worth adding to L1 track firmware output.
  // Outputting constraint chi2 easier than constrained helix params.).
  void beamConstraint(float& phi_con, float& rInv_con, float& pt_con, float& chi2XY_con, float& chi2XY_dof_con) const;

  /// set new Bfield
  void setBField(double aBField);

  /// Set bits in 96-bit Track word that corresponds to TFP firwmare output.
  void setTrackWordBits();

  std::string print() const;

  // Converter - 1/rinv measured in cm.
  double rinvToPt(double rinv) const { return (cLight_ / 1.0e11) * std::abs(theBField_ / rinv); }

};  /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Empty constructor
template <typename T>
TTTrack<T>::TTTrack()
    : theMomentum_(0., 0., 0.),
      thePOCA_(0., 0., 0.),
      theRInv_(0.),
      thePhi_(0.),
      theTanL_(0.),
      theD0_(0.),
      theZ0_(0.),
      thePhiSector_(0),
      theEtaSector_(0),
      theStubPtConsistency_(0.),
      theChi2_(0.),
      theChi2_XY_(0.),
      theChi2_Z_(0.),
      theNumFitPars_(0),
      theTrkMVA1_(0.),
      theTrkMVA2_(0.),
      theTrkMVA3_(0.),
      theTrackSeedType_(0),
      theBField_(0.) {}

/// Default constructor
template <typename T>
TTTrack<T>::TTTrack(double Rinv,
                    double phi0,
                    double tanL,
                    double z0,
                    double d0,
                    double aChi2XY,
                    double aChi2Z,
                    double trkMVA1,
                    double trkMVA2,
                    double trkMVA3,
                    unsigned int aHitPattern,
                    unsigned int nPar,
                    double Bfield,
                    unsigned int phiSector,
                    unsigned int etaSector,
                    double chi2BendRed,
                    unsigned int seedType,
                    const CovMat& helixCovMat)
    : thePOCA_(d0 * sin(phi0), -d0 * cos(phi0), z0),
      theRInv_(Rinv),
      thePhi_(phi0),
      theTanL_(tanL),
      theD0_(d0),
      theZ0_(z0),
      thePhiSector_(phiSector),
      theEtaSector_(etaSector),
      theStubPtConsistency_(chi2BendRed),
      theChi2_(aChi2XY + aChi2Z),
      theChi2_XY_(aChi2XY),
      theChi2_Z_(aChi2Z),
      theNumFitPars_(nPar),
      theHitPattern_(aHitPattern),
      theTrkMVA1_(trkMVA1),
      theTrkMVA2_(trkMVA2),
      theTrkMVA3_(trkMVA3),
      theTrackSeedType_(seedType),
      theBField_(Bfield),
      theHelixCovMat_(helixCovMat) {
  double pT = rinvToPt(Rinv);
  theMomentum_ = GlobalVector(GlobalVector::Cylindrical(pT, phi0, pT * tanL));
}

/// Get helix params & chi2XY after constraining track to x=y=0.
template <typename T>
void TTTrack<T>::beamConstraint(
    float& phi_con, float& rInv_con, float& pt_con, float& chi2XY_con, float& chi2XY_dof_con) const {
  phi_con = this->phi();
  rInv_con = this->rInv();
  pt_con = this->pt();
  chi2XY_con = this->chi2XY();
  chi2XY_dof_con = this->chi2XYRed();

  if (theNumFitPars_ == 5) {
    // Calculated with Lagrange multipliers in approx that d0 is small.
    double d0 = this->d0();
    // To constrain to beam-spot (XB,YB) rather than (0,0), would need this,
    // in approx that XB,YB,D0 all small.
    // double d0 = this->d0() - (XB*sin(phi_con) - YB cos(phi_con));
    double lagrange = d0 / theHelixCovMat_[Hpar::D0][Hpar::D0];
    chi2XY_con += lagrange * d0;
    rInv_con -= lagrange * theHelixCovMat_[Hpar::D0][Hpar::INVR];
    phi_con -= lagrange * theHelixCovMat_[Hpar::D0][Hpar::PHI0];
    phi_con = reco::deltaPhi(phi_con, 0.);
    pt_con = this->rinvToPt(rInv_con);
    constexpr int nHelixParsRPhi = 2;  // with d0 = 0 constraint
    int dof = theStubRefs.size() - nHelixParsRPhi;
    chi2XY_dof_con = chi2XY_con / dof;
  }
}

/// set B field if need be
template <typename T>
void TTTrack<T>::setBField(double aBField) {
  theBField_ = aBField;
  // if, for some reason, we want to change the value of the B-Field, recompute momentum:
  double thePT = this->rinvToPt(theRInv_);
  theMomentum_ = GlobalVector(GlobalVector::Cylindrical(thePT, thePhi_, thePT * theTanL_));

  return;
}

/// Set bits in 96-bit Track word
template <typename T>
void TTTrack<T>::setTrackWordBits() {
  constexpr unsigned int valid = true;
  constexpr double mvaOther = 0.;

  // missing conversion of global phi to difference from sector center phi

  setTrackWord(valid,
               momentum(),
               POCA(),
               rInv(),
               chi2XYRed(),
               chi2ZRed(),
               chi2BendRed(),
               hitPattern(),
               trkMVA1(),
               mvaOther,
               phiSector());

  return;
}

template <typename T>
std::string TTTrack<T>::print() const {
  const std::string padding("\t");
  std::stringstream output;
  output << padding << " -- TTTrack --\n";
  // Compare original helix params with undigi(digi()) ones.
  output << "Comparing float vs undigi(digi(float))\n";
  output << "Rinv      = " << rInv() << " vs " << getRinv() << "\n";
  output << "Local phi = " << localPhi() << " vs " << getPhi() << "\n";
  output << "tanL      = " << tanL() << " vs " << getTanl() << "\n";
  output << "z0        = " << z0() << " vs " << getZ0() << "\n";
  output << "d0        = " << d0() << " vs " << getD0() << "\n";
  output << "chi2XYRed = " << chi2XYRed() << " vs " << getChi2RPhi() << "\n";
  output << "trkMVA1   = " << trkMVA1() << " vs " << getMVAQuality() << "\n";
  auto safeSqrt = [](float q) { return q >= 0 ? sqrt(q) : -sqrt(-q); };
  output << "sigma(z0) = " << safeSqrt(theHelixCovMat_[Hpar::Z0][Hpar::Z0]) << "\n";
  if (theNumFitPars_ == 5)
    output << "sigma(d0) = " << safeSqrt(theHelixCovMat_[Hpar::D0][Hpar::D0]) << "\n";

  output << "digi(L1 track) word = " << getTrackWord().to_string(16) << "\n";

  unsigned int iStub = 0;
  for (const auto& stubIter : theStubRefs) {
    output << padding << "stub: " << iStub++ << ", DetId: " << (stubIter->getDetId()).rawId() << '\n';
  }

  return output.str();
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const TTTrack<T>& aTTTrack) {
  return (os << aTTTrack.print());
}

#endif
