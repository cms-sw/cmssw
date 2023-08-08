/*! \class   TTTrack
 *  \brief   Class to store the L1 Track Trigger tracks
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
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

namespace tttrack {
  void errorSetTrackWordBits(unsigned int);
}

template <typename T>
class TTTrack : public TTTrack_TrackWord {
private:
  /// Data members
  std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > theStubRefs;
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
  double theBField_;     // needed for unpacking
  int theGTTLinkIndex_;  // stores the position within a fiber link inside GTT
  static constexpr unsigned int Npars4 = 4;
  static constexpr unsigned int Npars5 = 5;
  static constexpr float MagConstant =
      CLHEP::c_light / 1.0E3;  //constant is 0.299792458; who knew c_light was in mm/ns?

public:
  /// Constructors
  TTTrack();

  TTTrack(double aRinv,
          double aphi,
          double aTanLambda,
          double az0,
          double ad0,
          double aChi2,
          double trkMVA1,
          double trkMVA2,
          double trkMVA3,
          unsigned int aHitpattern,
          unsigned int nPar,
          double Bfield);

  TTTrack(double aRinv,
          double aphi,
          double aTanLambda,
          double az0,
          double ad0,
          double aChi2xyfit,
          double aChi2zfit,
          double trkMVA1,
          double trkMVA2,
          double trkMVA3,
          unsigned int aHitpattern,
          unsigned int nPar,
          double Bfield);

  /// Destructor
  ~TTTrack();

  /// Track components
  std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > getStubRefs() const { return theStubRefs; }
  void addStubRef(edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > aStub) { theStubRefs.push_back(aStub); }
  void setStubRefs(std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > aStubs) {
    theStubRefs = aStubs;
  }

  /// Track momentum
  GlobalVector momentum() const;

  /// Track curvature
  double rInv() const;

  /// Track phi
  double phi() const;

  /// Local track phi (within the sector)
  double localPhi() const;

  /// Track tanL
  double tanL() const;

  /// Track d0
  double d0() const;

  /// Track z0
  double z0() const;

  /// Track eta
  double eta() const;

  /// POCA
  GlobalPoint POCA() const;

  /// MVA Track quality variables
  double trkMVA1() const;
  void settrkMVA1(double atrkMVA1);
  double trkMVA2() const;
  void settrkMVA2(double atrkMVA2);
  double trkMVA3() const;
  void settrkMVA3(double atrkMVA3);

  /// Phi Sector
  unsigned int phiSector() const { return thePhiSector_; }
  void setPhiSector(unsigned int aSector) { thePhiSector_ = aSector; }

  /// Eta Sector
  unsigned int etaSector() const { return theEtaSector_; }
  void setEtaSector(unsigned int aSector) { theEtaSector_ = aSector; }

  /// GTT Link Information
  unsigned int gttLinkID() const { return (eta() >= 0 ? 1 : 0) + (2 * phiSector()); }
  int gttLinkIndex() const { return theGTTLinkIndex_; }
  void setGTTLinkIndex(int idx) { theGTTLinkIndex_ = idx; }

  /// Track seeding (for debugging)
  unsigned int trackSeedType() const { return theTrackSeedType_; }
  void setTrackSeedType(int aSeed) { theTrackSeedType_ = aSeed; }

  /// Chi2
  double chi2() const;
  double chi2Red() const;
  double chi2Z() const;
  double chi2ZRed() const;
  double chi2XY() const;
  double chi2XYRed() const;

  /// Stub Pt consistency (i.e. stub bend chi2/dof)
  /// Note: The "stubPtConsistency" names are historic and people are encouraged
  ///       to adopt the "chi2Bend" names.
  double stubPtConsistency() const;
  void setStubPtConsistency(double aPtConsistency);
  double chi2BendRed() { return stubPtConsistency(); }
  void setChi2BendRed(double aChi2BendRed) { setStubPtConsistency(aChi2BendRed); }
  double chi2Bend() { return chi2BendRed() * theStubRefs.size(); }

  void setFitParNo(unsigned int aFitParNo);
  int nFitPars() const { return theNumFitPars_; }

  /// Hit Pattern
  unsigned int hitPattern() const;

  /// set new Bfield
  void setBField(double aBField);

  void setTrackWordBits();
  void testTrackWordBits();

  /// Information
  std::string print(unsigned int i = 0) const;

};  /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Default Constructor
template <typename T>
TTTrack<T>::TTTrack() {
  theStubRefs.clear();
  theMomentum_ = GlobalVector(0.0, 0.0, 0.0);
  theRInv_ = 0.0;
  thePOCA_ = GlobalPoint(0.0, 0.0, 0.0);
  theD0_ = 0.;
  theZ0_ = 0.;
  theTanL_ = 0;
  thePhi_ = 0;
  theTrkMVA1_ = 0;
  theTrkMVA2_ = 0;
  theTrkMVA3_ = 0;
  thePhiSector_ = 0;
  theEtaSector_ = 0;
  theTrackSeedType_ = 0;
  theGTTLinkIndex_ = -1;
  theChi2_ = 0.0;
  theChi2_XY_ = 0.0;
  theChi2_Z_ = 0.0;
  theStubPtConsistency_ = 0.0;
  theNumFitPars_ = 0;
}

/// Meant to be default constructor
template <typename T>
TTTrack<T>::TTTrack(double aRinv,
                    double aphi0,
                    double aTanlambda,
                    double az0,
                    double ad0,
                    double aChi2,
                    double trkMVA1,
                    double trkMVA2,
                    double trkMVA3,
                    unsigned int aHitPattern,
                    unsigned int nPar,
                    double aBfield) {
  theStubRefs.clear();
  double thePT = std::abs(MagConstant / aRinv * aBfield / 100.0);  // Rinv is in cm-1
  theMomentum_ = GlobalVector(GlobalVector::Cylindrical(thePT, aphi0, thePT * aTanlambda));
  theRInv_ = aRinv;
  thePOCA_ = GlobalPoint(ad0 * sin(aphi0), -ad0 * cos(aphi0), az0);
  theD0_ = ad0;
  theZ0_ = az0;
  thePhi_ = aphi0;
  theTanL_ = aTanlambda;
  thePhiSector_ = 0;      // must be set externally
  theEtaSector_ = 0;      // must be set externally
  theTrackSeedType_ = 0;  // must be set externally
  theGTTLinkIndex_ = -1;  // must be set externally
  theChi2_ = aChi2;
  theTrkMVA1_ = trkMVA1;
  theTrkMVA2_ = trkMVA2;
  theTrkMVA3_ = trkMVA3;
  theStubPtConsistency_ = 0.0;  // must be set externally
  theNumFitPars_ = nPar;
  theHitPattern_ = aHitPattern;
  theBField_ = aBfield;
  theChi2_XY_ = -999.;
  theChi2_Z_ = -999.;
}

/// Second default constructor with split chi2
template <typename T>
TTTrack<T>::TTTrack(double aRinv,
                    double aphi0,
                    double aTanlambda,
                    double az0,
                    double ad0,
                    double aChi2XY,
                    double aChi2Z,
                    double trkMVA1,
                    double trkMVA2,
                    double trkMVA3,
                    unsigned int aHitPattern,
                    unsigned int nPar,
                    double aBfield)
    : TTTrack(aRinv,
              aphi0,
              aTanlambda,
              az0,
              ad0,
              aChi2XY + aChi2Z,  // add chi2 values
              trkMVA1,
              trkMVA2,
              trkMVA3,
              aHitPattern,
              nPar,
              aBfield) {
  this->theChi2_XY_ = aChi2XY;
  this->theChi2_Z_ = aChi2Z;
}

/// Destructor
template <typename T>
TTTrack<T>::~TTTrack() {}

template <typename T>
void TTTrack<T>::setFitParNo(unsigned int nPar) {
  theNumFitPars_ = nPar;

  return;
}

// Note that these calls return the floating point values.  If a TTTrack is made with only ditized values,
// the unpacked values must come from the TTTrack_Trackword member functions.

template <typename T>
GlobalVector TTTrack<T>::momentum() const {
  return theMomentum_;
}

template <typename T>
double TTTrack<T>::rInv() const {
  return theRInv_;
}

template <typename T>
double TTTrack<T>::tanL() const {
  return theTanL_;
}

template <typename T>
double TTTrack<T>::eta() const {
  return theMomentum_.eta();
}

template <typename T>
double TTTrack<T>::phi() const {
  return thePhi_;
}

template <typename T>
double TTTrack<T>::localPhi() const {
  return TTTrack_TrackWord::localPhi(thePhi_, thePhiSector_);
}

template <typename T>
double TTTrack<T>::d0() const {
  return theD0_;
}

template <typename T>
double TTTrack<T>::z0() const {
  return theZ0_;
}

template <typename T>
GlobalPoint TTTrack<T>::POCA() const {
  return thePOCA_;
}

/// Chi2
template <typename T>
double TTTrack<T>::chi2() const {
  return theChi2_;
}

/// Chi2Z
template <typename T>
double TTTrack<T>::chi2Z() const {
  return theChi2_Z_;
}

/// Chi2XY
template <typename T>
double TTTrack<T>::chi2XY() const {
  return theChi2_XY_;
}

/// Chi2 reduced
template <typename T>
double TTTrack<T>::chi2Red() const {
  return theChi2_ / (2 * theStubRefs.size() - theNumFitPars_);
}

/// Chi2XY reduced
template <typename T>
double TTTrack<T>::chi2XYRed() const {
  return theChi2_XY_ / (theStubRefs.size() - (theNumFitPars_ - 2));
}

/// Chi2Z reduced
template <typename T>
double TTTrack<T>::chi2ZRed() const {
  return theChi2_Z_ / (theStubRefs.size() - 2.);
}

template <typename T>
double TTTrack<T>::trkMVA1() const {
  return theTrkMVA1_;
}

template <typename T>
void TTTrack<T>::settrkMVA1(double atrkMVA1) {
  theTrkMVA1_ = atrkMVA1;
  return;
}

template <typename T>
double TTTrack<T>::trkMVA2() const {
  return theTrkMVA2_;
}

template <typename T>
void TTTrack<T>::settrkMVA2(double atrkMVA2) {
  theTrkMVA2_ = atrkMVA2;
  return;
}

template <typename T>
double TTTrack<T>::trkMVA3() const {
  return theTrkMVA3_;
}

template <typename T>
void TTTrack<T>::settrkMVA3(double atrkMVA3) {
  theTrkMVA3_ = atrkMVA3;
  return;
}

/// StubPtConsistency
template <typename T>
void TTTrack<T>::setStubPtConsistency(double aStubPtConsistency) {
  theStubPtConsistency_ = aStubPtConsistency;
  return;
}

/// StubPtConsistency
template <typename T>
double TTTrack<T>::stubPtConsistency() const {
  return theStubPtConsistency_;
}

/// Hit Pattern
template <typename T>
unsigned int TTTrack<T>::hitPattern() const {
  return theHitPattern_;
}

/// set B field if need be
template <typename T>
void TTTrack<T>::setBField(double aBField) {
  // if, for some reason, we want to change the value of the B-Field, recompute pT and momentum:
  double thePT = std::abs(MagConstant / theRInv_ * aBField / 100.0);  // Rinv is in cm-1
  theMomentum_ = GlobalVector(GlobalVector::Cylindrical(thePT, thePhi_, thePT * theTanL_));

  return;
}

/// Set bits in 96-bit Track word
template <typename T>
void TTTrack<T>::setTrackWordBits() {
  if (!(theNumFitPars_ == Npars4 || theNumFitPars_ == Npars5)) {
    tttrack::errorSetTrackWordBits(theNumFitPars_);
    return;
  }

  unsigned int valid = true;
  unsigned int mvaQuality = 0;
  unsigned int mvaOther = 0;

  // missing conversion of global phi to difference from sector center phi

  if (theChi2_Z_ < 0) {
    setTrackWord(valid,
                 theMomentum_,
                 thePOCA_,
                 theRInv_,
                 theChi2_,
                 0,
                 theStubPtConsistency_,
                 theHitPattern_,
                 mvaQuality,
                 mvaOther,
                 thePhiSector_);
  } else {
    setTrackWord(valid,
                 theMomentum_,
                 thePOCA_,
                 theRInv_,
                 chi2XYRed(),
                 chi2ZRed(),
                 chi2BendRed(),
                 theHitPattern_,
                 mvaQuality,
                 mvaOther,
                 thePhiSector_);
  }
  return;
}

/// Test bits in 96-bit Track word
template <typename T>
void TTTrack<T>::testTrackWordBits() {
  //  float rPhi = theMomentum_.phi();  // this needs to be phi relative to center of sector ****
  //float rEta = theMomentum_.eta();
  //float rZ0 = thePOCA_.z();
  //float rD0 = thePOCA_.perp();

  //this is meant for debugging only.

  //std::cout << " phi " << rPhi << " " << get_iphi() << std::endl;
  //std::cout << " eta " << rEta << " " << get_ieta() << std::endl;
  //std::cout << " Z0 " << rZ0 << " " << get_iz0() << std::endl;
  //std::cout << " D0 " << rD0 << " " << get_id0() << std::endl;
  //std::cout << " Rinv " << theRInv_ << " " << get_iRinv() << std::endl;
  //std::cout << " chi2 " << theChi2_ << " " << get_ichi2() << std::endl;

  return;
}

/// Information
template <typename T>
std::string TTTrack<T>::print(unsigned int i) const {
  std::string padding("");
  for (unsigned int j = 0; j != i; ++j) {
    padding += "\t";
  }

  std::stringstream output;
  output << padding << "TTTrack:\n";
  padding += '\t';
  output << '\n';
  unsigned int iStub = 0;

  typename std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > >::const_iterator stubIter;
  for (stubIter = theStubRefs.begin(); stubIter != theStubRefs.end(); ++stubIter) {
    output << padding << "stub: " << iStub++ << ", DetId: " << ((*stubIter)->getDetId()).rawId() << '\n';
  }

  return output.str();
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const TTTrack<T>& aTTTrack) {
  return (os << aTTTrack.print());
}

#endif
