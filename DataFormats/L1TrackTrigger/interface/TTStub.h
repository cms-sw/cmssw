/*! \class   TTStub
 *  \brief   Class to store the L1 Track Trigger stubs
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than Phase2TrackerDigis
 *           in case there is such a need in the future.
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_FORMAT_H
#define L1_TRACK_TRIGGER_STUB_FORMAT_H

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

template <typename T>
class TTStub {
public:
  /// Constructors
  TTStub();
  TTStub(DetId aDetId);

  /// Destructor
  ~TTStub();

  /// Data members:   aBc( ... )
  /// Helper methods: findAbc( ... )

  /// Clusters composing the Stub -- see https://twiki.cern.ch/twiki/bin/viewauth/CMS/SLHCTrackerTriggerSWTools#TTCluster

  /// Returns the permanent references of the cluster in the sensor stack identified by hitStackMember
  /// which should be either 0 or 1 for the innermost and outermost sensor, respectively
  const edm::Ref<edmNew::DetSetVector<TTCluster<T> >, TTCluster<T> >& clusterRef(unsigned int hitStackMember) const;

  /// Add a cluster reference, depending on which stack member it is on (inner = 0, outer = 1)
  void addClusterRef(edm::Ref<edmNew::DetSetVector<TTCluster<T> >, TTCluster<T> > aTTCluster);

  /// Detector element
  DetId getDetId() const { return theDetId; }
  void setDetId(DetId aDetId) { theDetId = aDetId; }

  /// Trigger information; see, e.g., TTStubAlgorithm_official::PatternHitCorrelation for definitions
  /// Values are passed back from there via the TTStubBuilder, be careful to choose the right one.
  /// In particular note the difference between values passed as "full" strip units (i.e. the strip
  /// number or difference between strip numbers) and "half" strip units, which have a 2X finer granularity.

  /// RawBend() [rename of getTriggerDisplacement()]: In FULL strip units!
  /// Returns the relative displacement between the two cluster centroids, i.e.
  /// the difference between average row coordinates in inner and outer stack member,
  /// in terms of outer member pitch (if both pitches are the same, this is just the coordinate difference).
  /// Flag for rejected stubs: +500 if rejected by FE, +1000 if rejected by CIC chip.
  double rawBend() const;

  /// setRawBend [rename of setTriggerDisplacement()]: In HALF strip units!
  /// Sets relative displacement between the two cluster centroids, as above.
  /// Flag for rejected stubs: +500 if rejected by FE, +1000 if rejected by CIC chip.
  /// NB: Should probably only be used in TTStubBuilder or very similar.
  void setRawBend(int aDisplacement);

  /// BendOffset() [rename of getTriggerOffset()]: In FULL strip units!
  /// Returns the correction offset calculated while accepting/rejecting the stub
  /// Offset is the projection of a straight line from the beam-spot through the innermost hit
  /// to the outermost stack member, again in terms of outer member pitch.  It is calculated
  /// taking the centre of the module at (NROWS/2)-0.5.

  double bendOffset() const;

  /// setBendOffset() [rename of setTriggerOffset()]: In HALF strip units!
  /// Again restricted to builder code.
  void setBendOffset(int anOffset);

  /// set whether this is a PS module or not;
  void setModuleTypePS(bool isPSModule);

  /// check if a PS module
  bool moduleTypePS() const;

  /// CBC3-style trigger information
  /// for sake of simplicity, these methods are
  /// slightly out of the ABC(...)/findABC(...) rule

  /// InnerClusterPosition() [rename of getTriggerPosition()]: In FULL strip units!
  /// Returns the average local x coordinate of hits in the inner stack member
  double innerClusterPosition() const;

  /// BendFE(): In FULL-STRIP units from FE! Rename of getTriggerBend()
  double bendFE() const;

  /// BendBE(): In FULL-STRIP units! Reduced resolution from BE boards.  Rename of getHardwareBend()
  double bendBE() const;

  ///  setBendBE(): In HALF-STRIP units!  Reduced resolution in BE boards.  Rename of setHardwareBend()
  void setBendBE(float aBend);

  /// Print Stub information for debugging purposes
  std::string print(unsigned int i = 0) const;

private:
  /// Data members
  DetId theDetId;
  edm::Ref<edmNew::DetSetVector<TTCluster<T> >, TTCluster<T> > theClusterRef0;
  edm::Ref<edmNew::DetSetVector<TTCluster<T> >, TTCluster<T> > theClusterRef1;
  int theDisplacement;
  int theOffset;
  float theBendBE;
  bool thePSModule;

  static constexpr float dummyBend = 999999;  // Dumy value should be away from potential bends
};                                            /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Default Constructor
template <typename T>
TTStub<T>::TTStub() {
  /// Set default data members
  theDetId = 0;
  theDisplacement = dummyBend;
  theOffset = 0;
  theBendBE = dummyBend;
}

/// Another Constructor using a given DetId
template <typename T>
TTStub<T>::TTStub(DetId aDetId) {
  /// Set data members
  this->setDetId(aDetId);

  /// Set default data members
  theDisplacement = dummyBend;
  theOffset = 0;
  theBendBE = dummyBend;
}

/// Destructor
template <typename T>
TTStub<T>::~TTStub() {}

template <typename T>
const edm::Ref<edmNew::DetSetVector<TTCluster<T> >, TTCluster<T> >& TTStub<T>::clusterRef(
    unsigned int hitStackMember) const {
  return (hitStackMember == 0) ? theClusterRef0 : theClusterRef1;
}

template <typename T>
void TTStub<T>::addClusterRef(edm::Ref<edmNew::DetSetVector<TTCluster<T> >, TTCluster<T> > aTTCluster) {
  if (aTTCluster->getStackMember() == 0)
    theClusterRef0 = aTTCluster;
  else if (aTTCluster->getStackMember() == 1)
    theClusterRef1 = aTTCluster;
}

/// Trigger info
template <typename T>
double TTStub<T>::rawBend() const {
  return 0.5 * theDisplacement;
}

template <typename T>
void TTStub<T>::setRawBend(int aDisplacement) {
  theDisplacement = aDisplacement;
}

template <typename T>
double TTStub<T>::bendOffset() const {
  return 0.5 * theOffset;
}

template <typename T>
void TTStub<T>::setBendOffset(int anOffset) {
  theOffset = anOffset;
}

template <typename T>
void TTStub<T>::setBendBE(float aBend) {
  theBendBE = aBend;
}

template <typename T>
void TTStub<T>::setModuleTypePS(bool isPSModule) {
  thePSModule = isPSModule;
}  // set whether this is a PS module or not;
template <typename T>
bool TTStub<T>::moduleTypePS() const {
  return thePSModule;
}  // check if a PS module
template <typename T>
double TTStub<T>::innerClusterPosition() const {
  return this->clusterRef(0)->findAverageLocalCoordinates().x();  //CBC3-style trigger info
}

template <typename T>
double TTStub<T>::bendFE() const {
  if (theDisplacement == dummyBend)
    return theDisplacement;

  return 0.5 * (theDisplacement - theOffset);
}

template <typename T>
double TTStub<T>::bendBE() const {
  if (theBendBE == dummyBend)
    return this->bendFE();  // If not set make it transparent

  return theBendBE;
}

/// Information
template <typename T>
std::string TTStub<T>::print(unsigned int i) const {
  std::string padding("");
  for (unsigned int j = 0; j != i; ++j) {
    padding += "\t";
  }

  std::stringstream output;
  output << padding << "TTStub:\n";
  padding += '\t';
  output << padding << "DetId: " << theDetId.rawId() << ", position: " << this->InnerClusterPosition();
  output << ", bend: " << this->BendFE() << '\n';
  output << ", hardware bend: " << this->BendBE() << '\n';
  output << padding << "cluster 0: address: " << theClusterRef0.get();
  output << ", cluster size: " << theClusterRef0->getHits().size() << '\n';
  output << padding << "cluster 1: address: " << theClusterRef1.get();
  output << ", cluster size: " << theClusterRef1->getHits().size() << '\n';
  return output.str();
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const TTStub<T>& aTTStub) {
  return (os << aTTStub.print());
}

#endif
