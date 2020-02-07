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

#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"

template <typename T>
class TTTrack : public TTTrack_TrackWord {
private:
  /// Data members
  std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > theStubRefs;
  GlobalVector theMomentum;
  GlobalPoint thePOCA;
  double theRInv;
  double thePhi;
  double theTanL;
  double theD0;
  double theZ0;
  unsigned int thePhiSector;
  unsigned int theEtaSector;
  double theStubPtConsistency;
  double theChi2;
  double theChi2XY;
  double theChi2Z;
  unsigned int numFitPars;
  unsigned int theHitPattern;
  double theTrkMVA1;
  double theTrkMVA2;
  double theTrkMVA3;
  int theTrackSeedType;
  double theBField;  // needed for unpacking
  static constexpr unsigned int Npars4 = 4;
  static constexpr unsigned int Npars5 = 5;
  static constexpr float MagConstant = 0.299792458;

public:
  /// Constructors
  TTTrack();
  TTTrack(std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > aStubs);

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
  double trkMVA2() const;
  double trkMVA3() const;

  /// Phi Sector
  unsigned int phiSector() const { return thePhiSector; }
  void setPhiSector(unsigned int aSector) { thePhiSector = aSector; }

  /// Eta Sector
  unsigned int etaSector() const { return theEtaSector; }
  void setEtaSector(unsigned int aSector) { theEtaSector = aSector; }

  /// Track seeding (for debugging)
  unsigned int trackSeedType() const { return theTrackSeedType; }
  void setTrackSeedType(int aSeed) { theTrackSeedType = aSeed; }

  /// Chi2
  double chi2() const;
  double chi2Red() const;
  double chi2z() const;

  /// Stub Pt consistency
  double stubPtConsistency() const;
  void setStubPtConsistency(double aPtConsistency);

  void setFitParNo(unsigned int aFitParNo);
  int nFitPars() const { return numFitPars; }

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
  theMomentum = GlobalVector(0.0, 0.0, 0.0);
  theRInv = 0.0;
  thePOCA = GlobalPoint(0.0, 0.0, 0.0);
  theD0 = 0.;
  theZ0 = 0.;
  theTanL = 0;
  thePhi = 0;
  theTrkMVA1 = 0;
  theTrkMVA2 = 0;
  theTrkMVA3 = 0;
  thePhiSector = 0;
  theEtaSector = 0;
  theTrackSeedType = 0;
  theChi2 = 0.0;
  theStubPtConsistency = 0.0;
  numFitPars = 0;
}

/// Another Constructor
template <typename T>
TTTrack<T>::TTTrack(std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > aStubs) {
  theStubRefs = aStubs;
  theMomentum = GlobalVector(0.0, 0.0, 0.0);
  theRInv = 0.0;
  thePOCA = GlobalPoint(0.0, 0.0, 0.0);
  theD0 = 0.;
  theZ0 = 0.;
  theTanL = 0;
  thePhi = 0;
  theTrkMVA1 = 0;
  theTrkMVA2 = 0;
  theTrkMVA3 = 0;
  thePhiSector = 0;
  theEtaSector = 0;
  theTrackSeedType = 0;
  theChi2 = 0.0;
  theStubPtConsistency = 0.0;
  numFitPars = 0;
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
  double thePT = fabs(MagConstant * 1.0 / aRinv * aBfield / 100.0);  // Rinv is in cm-1
  theMomentum = GlobalVector(GlobalVector::Cylindrical(thePT, aphi0, thePT * aTanlambda));
  theRInv = aRinv;
  thePOCA = GlobalPoint(ad0 * cos(aphi0), ad0 * sin(aphi0), az0);
  theD0 = ad0;
  theZ0 = az0;
  thePhi = aphi0;
  theTanL = aTanlambda;
  thePhiSector = 0;      // must be set externally
  theEtaSector = 0;      // must be set externally
  theTrackSeedType = 0;  // must be set externally
  theChi2 = aChi2;
  theTrkMVA1 = trkMVA1;
  theTrkMVA2 = trkMVA2;
  theTrkMVA3 = trkMVA3;
  theStubPtConsistency = 0.0;  // must be set externally
  numFitPars = nPar;
  theHitPattern = aHitPattern;
  theBField = aBfield;
  // should probably fill the momentum vectur
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
                    double aBfield) {
  theStubRefs.clear();
  double thePT = fabs(MagConstant * 1.0 / aRinv * aBfield / 100.0);  // Rinv is in cm-1
  theMomentum = GlobalVector(GlobalVector::Cylindrical(thePT, aphi0, thePT * aTanlambda));
  theRInv = aRinv;
  thePOCA = GlobalPoint(ad0 * cos(aphi0), ad0 * sin(aphi0), az0);
  theD0 = ad0;
  theZ0 = az0;
  thePhi = aphi0;
  theTanL = aTanlambda;
  thePhiSector = 0;      // must be set externally
  theEtaSector = 0;      // must be set externally
  theTrackSeedType = 0;  // must be set externally
  theChi2XY = aChi2XY;
  theChi2Z = aChi2Z;
  theTrkMVA1 = trkMVA1;
  theTrkMVA2 = trkMVA2;
  theTrkMVA3 = trkMVA3;
  theStubPtConsistency = 0.0;  // must be set externally
  numFitPars = nPar;
  theHitPattern = aHitPattern;
  theBField = aBfield;
  // should probably fill the momentum vectur
}

/// Destructor
template <typename T>
TTTrack<T>::~TTTrack() {}

template <typename T>
void TTTrack<T>::setFitParNo(unsigned int nPar) {
  numFitPars = nPar;

  return;
}

// Note that these calls return the floating point values.  If a TTTrack is made with only ditized values,
// the unpacked values must come from the TTTrack_Trackword member functions.

template <typename T>
GlobalVector TTTrack<T>::momentum() const {
  return theMomentum;
}

template <typename T>
double TTTrack<T>::rInv() const {
  return theRInv;
}

template <typename T>
double TTTrack<T>::tanL() const {
  return theTanL;
}

template <typename T>
double TTTrack<T>::eta() const {
  return theMomentum.eta();
}

template <typename T>
double TTTrack<T>::phi() const {
  return thePhi;
}

template <typename T>
double TTTrack<T>::d0() const {
  return theD0;
}

template <typename T>
double TTTrack<T>::z0() const {
  return theZ0;
}

template <typename T>
GlobalPoint TTTrack<T>::POCA() const {
  return thePOCA;
}

/// Chi2
template <typename T>
double TTTrack<T>::chi2() const {
  return theChi2;
}

/// Chi2z
template <typename T>
double TTTrack<T>::chi2z() const {
  return theChi2Z;
}

/// Chi2 reduced
template <typename T>
double TTTrack<T>::chi2Red() const {
  return theChi2 / (2 * theStubRefs.size() - numFitPars);
}

/// MVA quality variables
template <typename T>
double TTTrack<T>::trkMVA1() const {
  return theTrkMVA1;
}

template <typename T>
double TTTrack<T>::trkMVA2() const {
  return theTrkMVA2;
}

template <typename T>
double TTTrack<T>::trkMVA3() const {
  return theTrkMVA3;
}

/// StubPtConsistency
template <typename T>
void TTTrack<T>::setStubPtConsistency(double aStubPtConsistency) {
  theStubPtConsistency = aStubPtConsistency;
  return;
}

/// StubPtConsistency
template <typename T>
double TTTrack<T>::stubPtConsistency() const {
  return theStubPtConsistency;
}

/// Set bits in 96-bit Track word
template <typename T>
void TTTrack<T>::setTrackWordBits() {
  if (!(numFitPars == Npars4 || numFitPars == Npars5)) {
    edm::LogError("TTTrack") << " setTrackWordBits method is called with numFitPars=" << numFitPars
                             << " only possible values are 4/5" << std::endl;
    return;
  }

  unsigned int sparebits = 0;

  // missing conversion of global phi to difference from sector center phi

  if (theChi2Z == 0.) {
    setTrackWord(theMomentum, thePOCA, theRInv, theChi2, theChi2Z, theStubPtConsistency, theHitPattern, sparebits);

  } else {
    setTrackWord(theMomentum, thePOCA, theRInv, theChi2XY, theChi2Z, theStubPtConsistency, theHitPattern, sparebits);
  }
  return;
}

/// Test bits in 96-bit Track word
template <typename T>
void TTTrack<T>::testTrackWordBits() {
  //  float rPhi = theMomentum.phi();  // this needs to be phi relative to center of sector ****
  //float rEta = theMomentum.eta();
  //float rZ0 = thePOCA.z();
  //float rD0 = thePOCA.perp();

  //this is meant for debugging only.

  //std::cout << " phi " << rPhi << " " << get_iphi() << std::endl;
  //std::cout << " eta " << rEta << " " << get_ieta() << std::endl;
  //std::cout << " Z0 " << rZ0 << " " << get_iz0() << std::endl;
  //std::cout << " D0 " << rD0 << " " << get_id0() << std::endl;
  //std::cout << " Rinv " << theRInv << " " << get_iRinv() << std::endl;
  //std::cout << " chi2 " << theChi2 << " " << get_ichi2() << std::endl;

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
