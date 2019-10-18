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
  double theEta;
  double theD0;
  double theZ0;
  unsigned int thePhiSector;
  double theStubPtConsistency;
  double theChi2;
  unsigned int NumFitPars;
  unsigned int theHitPattern;
  float an_MVA_Value;
  float another_MVA_Value;
  int theTrackSeed;
  double theBField;  // needed for unpacking
  static constexpr unsigned int Npars4 = 4;
  static constexpr unsigned int Npars5 = 5;
  static constexpr float MagConstant = 0.3;

public:
  /// Constructors
  TTTrack();
  TTTrack(std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > aStubs);

  TTTrack(double aRinv,
          double aphi,
          double aeta,
          double az0,
          double ad0,
          double aChi2,
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
  GlobalVector getMomentum(unsigned int npar = Npars4) const;

  /// Track curvature
  double rInv() const;
  double getRInv(unsigned int npar = Npars4) const;

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
  GlobalPoint getPOCA(unsigned int npar = Npars4) const;

  /// Phi Sector
  unsigned int PhiSector() const { return thePhiSector; }
  unsigned int getSector() const { return thePhiSector; }
  void setPhiSector(unsigned int aSector) { thePhiSector = aSector; }

  /// Track seeding (for debugging)
  unsigned int TrackSeed() const { return theTrackSeed; }
  void setTrackSeed(int aSeed) { theTrackSeed = aSeed; }

  /// Chi2
  double chi2() const;
  double chi2Red() const;
  double getChi2(unsigned int npar = 4) const;
  double getChi2Red(unsigned int npar = 4) const;

  /// Stub Pt consistency
  double getStubPtConsistency(unsigned int npar = 4) const;
  void setStubPtConsistency(double aPtConsistency);

  void setFitParNo(unsigned int aFitParNo);

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
  thePhiSector = 0;
  theTrackSeed = 0;
  theChi2 = 0.0;
  theStubPtConsistency = 0.0;
  NumFitPars = 0;
}

/// Another Constructor
template <typename T>
TTTrack<T>::TTTrack(std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > aStubs) {
  theStubRefs = aStubs;
  theMomentum = GlobalVector(0.0, 0.0, 0.0);
  theRInv = 0.0;
  thePOCA = GlobalPoint(0.0, 0.0, 0.0);
  thePhiSector = 0;
  theTrackSeed = 0;
  theChi2 = 0.0;
  theStubPtConsistency = 0.0;
  NumFitPars = 0;
}

/// Meant to be default constructor
template <typename T>
TTTrack<T>::TTTrack(double aRinv,
                    double aphi,
                    double aeta,
                    double az0,
                    double ad0,
                    double aChi2,
                    unsigned int aHitPattern,
                    unsigned int nPar,
                    double aBfield) {
  theStubRefs.clear();
  double thePT = MagConstant * aRinv * aBfield;
  theMomentum = GlobalVector(GlobalVector::Cylindrical(thePT, aphi, thePT * sinh(aeta)));
  theRInv = aRinv;
  thePOCA = GlobalPoint(ad0 * cos(aphi), ad0 * sin(aphi), az0);
  thePhiSector = 0;  // must be set externally
  theTrackSeed = 0;  // must be set externally
  theChi2 = aChi2;
  theStubPtConsistency = 0.0;  // must be set externally
  NumFitPars = nPar;
  theHitPattern = aHitPattern;
  theBField = aBfield;
  // should probably fill the momentum vectur
}

/// Destructor
template <typename T>
TTTrack<T>::~TTTrack() {}

template <typename T>
void TTTrack<T>::setFitParNo(unsigned int nPar) {
  NumFitPars = nPar;

  return;
}

// Note that these calls return the floating point values.  If a TTTrack is made with only ditized values,
// the unpacked values must come from the TTTrack_Trackword member functions.

template <typename T>
GlobalVector TTTrack<T>::momentum() const {
  if (NumFitPars == Npars5 || NumFitPars == Npars4) {
    return theMomentum;
  } else
    return GlobalVector(0.0, 0.0, 0.0);
}

template <typename T>
GlobalVector TTTrack<T>::getMomentum(unsigned int npar) const {
  return momentum();
}

template <typename T>
double TTTrack<T>::rInv() const {
  if (NumFitPars == Npars5 || NumFitPars == Npars4) {
    return theRInv;
  } else
    return 0.0;
}

template <typename T>
double TTTrack<T>::getRInv(unsigned int npar) const {  //backwards compatibility

  return rInv();
}

template <typename T>
double TTTrack<T>::tanL() const {
  if (NumFitPars == Npars5 || NumFitPars == Npars4) {
    return theTanL;
  } else
    return 0.0;
}

template <typename T>
double TTTrack<T>::eta() const {
  if (NumFitPars == Npars5 || NumFitPars == Npars4) {
    return theEta;
  } else
    return 0.0;
}

template <typename T>
double TTTrack<T>::phi() const {
  if (NumFitPars == Npars5 || NumFitPars == Npars4) {
    return thePhi;
  } else
    return 0.0;
}

template <typename T>
double TTTrack<T>::d0() const {
  if (NumFitPars == Npars5 || NumFitPars == Npars4) {
    return theD0;
  } else
    return 0.0;
}

template <typename T>
double TTTrack<T>::z0() const {
  if (NumFitPars == Npars5 || NumFitPars == Npars4) {
    return theZ0;
  } else
    return 0.0;
}

template <typename T>
GlobalPoint TTTrack<T>::POCA() const {
  if (NumFitPars == Npars5 || NumFitPars == Npars4) {
    return thePOCA;
  } else
    return GlobalPoint(0.0, 0.0, 0.0);
}

template <typename T>
GlobalPoint TTTrack<T>::getPOCA(unsigned int npar) const  //backwards compatibility
{
  return POCA();
}

/// Chi2
template <typename T>
double TTTrack<T>::chi2() const {
  if (NumFitPars == Npars5 || NumFitPars == Npars4) {
    return theChi2;
  } else
    return 0.0;
}

template <typename T>
double TTTrack<T>::getChi2(unsigned int npar) const  //backwards compatibility
{
  return chi2();
}

/// Chi2 reduced
template <typename T>
double TTTrack<T>::chi2Red() const {
  if (NumFitPars == Npars5 || NumFitPars == Npars4) {
    return theChi2 / (2 * theStubRefs.size() - NumFitPars);
  } else
    return 0.0;
}

template <typename T>
double TTTrack<T>::getChi2Red(unsigned int npar) const  //backwards compatibility
{
  return chi2Red();
}

/// StubPtConsistency
template <typename T>
void TTTrack<T>::setStubPtConsistency(double aStubPtConsistency) {
  theStubPtConsistency = aStubPtConsistency;

  return;
}

/// StubPtConsistency
template <typename T>
double TTTrack<T>::getStubPtConsistency(unsigned int npar) const {
  if (NumFitPars == Npars5 || NumFitPars == Npars4) {
    return theStubPtConsistency;
  } else
    return 0.0;
}

/// Set bits in 96-bit Track word
template <typename T>
void TTTrack<T>::setTrackWordBits() {
  if (!(NumFitPars == Npars4 || NumFitPars == Npars5)) {
    edm::LogError("TTTrack") << " setTrackWordBits method is called with NumFitPars=" << NumFitPars
                             << " only possible values are 4/5" << std::endl;
    return;
  }

  unsigned int sparebits = 0;

  // missing conversion of global phi to difference from sector center phi

  setTrackWord(theMomentum, thePOCA, theRInv, theChi2, theStubPtConsistency, theHitPattern, sparebits);

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
