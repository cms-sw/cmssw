/*! \class   TTCluster
 *  \brief   Class to store the L1 Track Trigger clusters
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than Phase2TrackerDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \author Emmanuele Salvati
 *  \date   2013, Jul 12
 *
 *  Simplified Ian Tomalin (2026)
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_FORMAT_H
#define L1_TRACK_TRIGGER_CLUSTER_FORMAT_H

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"  /// NOTE: this is needed even if it seems not

template <typename T>
class TTCluster {
public:
  /// Constructors
  TTCluster();
  TTCluster(const std::vector<T>& aHits, const DetId& aDetId, unsigned int aStackMember);
  TTCluster(const DetId& aDetId, unsigned int aStackMember, const Phase2TrackerDigi& firstDigi, unsigned int width);

  /// Destructor
  ~TTCluster();

  /// Data members:   getABC( ... )
  /// Helper methods: findABC( ... )

  /// Detector module & which of two sensors inside it.
  const DetId& getDetId() const { return theDetId; }
  unsigned int getStackMember() const { return theStackMember; }

  // In CMSSW, (rows,cols) are in (r-phi,r-z), whereas FE chip spec doc has
  // opposite convention. These vectors have same size.
  const std::vector<int>& getRows() const { return theRows; }
  const std::vector<int>& getCols() const { return theCols; }

  unsigned int getNumHits() const { return theRows.size(); }

  /// Cluster width
  unsigned int findWidth() const;

  /// First row (i.e. in r-phi) in cluster.
  unsigned int findFirstRow() const;

  // Encoded channel number of hit j in cluster.
  uint16_t getChannel(unsigned int j) const {
    assert(j < theRows.size());
    // TO FIX: Take const here from Phase2TrackerDigi.
    return theRows[j] | theCols[j] << 10;
  }

  /// Individual hit (i.e. digi) coordinates (in units of pitch)
  MeasurementPoint findHitLocalCoordinates(unsigned int hitIdx) const;
  /// Twice inweighted centroid (in r-phi units of pitch)
  unsigned int     twiceCentroid() {return (2*findFirstRow() + findWidth() - 1);}
  /// Average cluster coordinates (in units of pitch)
  MeasurementPoint findAverageLocalCoordinates() const;
  MeasurementPoint findAverageLocalCoordinatesCentered() const;

  bool operator==(const TTCluster<T>& other) const;

  /// Information
  std::string print(unsigned int i = 0) const;

private:
  /// Set rows and columns to get rid of Digi collection
  /// Old code
  void setRowsCols(const std::vector<T>& aHits);
  /// New code -- knows clustering done only in r-phi
  void setRowsCols(const Phase2TrackerDigi& firstDigi, unsigned int width);

private:
  /// Data members
  DetId theDetId;
  unsigned int theStackMember;

  std::vector<int> theRows;
  std::vector<int> theCols;

};  /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Null Constructor
template <typename T>
TTCluster<T>::TTCluster() : theDetId(0), theStackMember(0) {}

/// Old Constructor
template <typename T>
TTCluster<T>::TTCluster(const std::vector<T>& aHits, const DetId& aDetId, unsigned int aStackMember)
    : theDetId(aDetId), theStackMember(aStackMember) {
  // Set theRows & theCols in cluster
  this->setRowsCols(aHits);
}

/// New Constructor
template <typename T>
TTCluster<T>::TTCluster(const DetId& aDetId,
                        unsigned int aStackMember,
                        const Phase2TrackerDigi& firstDigi,
                        unsigned int width)
    : theDetId(aDetId), theStackMember(aStackMember) {
  // Set theRows & theCols in cluster
  // FIX -- Replace theRows & theCols data members by firstDigi & width.
  this->setRowsCols(firstDigi, width);
}

/// Destructor
template <typename T>
TTCluster<T>::~TTCluster() {}

/// Cluster width
template <>
unsigned int TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>::findWidth() const;

/// Single hit coordinates
/// Average cluster coordinates
template <>
MeasurementPoint TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>::findHitLocalCoordinates(
    unsigned int hitIdx) const;

template <>
MeasurementPoint
TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>::findAverageLocalCoordinates() const;

template <typename T>
void TTCluster<T>::setRowsCols(const std::vector<T>& aHits) {}

template <>
void TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>::setRowsCols(
    const std::vector<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>& aHits);

/// Store coordinates locally -- New code -- knows clustering done only in r-phi
template <typename T>
void TTCluster<T>::setRowsCols(const Phase2TrackerDigi& firstDigi, unsigned int width) {
  theRows.reserve(width);
  theCols.reserve(width);

  unsigned int firstRow = firstDigi.row();
  unsigned int col = firstDigi.column();
  for (unsigned int i = 0; i < width; i++) {
    theRows.push_back(firstRow + i);
    // FIX: Replace theCols data member with single number.
    theCols.push_back(col);
  }
}

template <typename T>
bool TTCluster<T>::operator==(const TTCluster<T>& other) const {
  bool same = (theRows == other.getRows() && theCols == other.getCols() && theDetId == other.getDetId() &&
               theStackMember == other.getStackMember());
  return same;
}

/// Information
template <typename T>
std::string TTCluster<T>::print(unsigned int i) const {
  std::string padding("");
  for (unsigned int j = 0; j != i; ++j) {
    padding += "\t";
  }

  std::stringstream output;
  output << padding << "TTCluster:\n";
  padding += '\t';
  output << padding << "DetId: " << theDetId.rawId() << '\n';
  output << padding << "member: " << theStackMember << ", cluster size: " << this->getNumHits() << '\n';
  return output.str();
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const TTCluster<T>& aTTCluster) {
  return (os << aTTCluster.print());
}

#endif
