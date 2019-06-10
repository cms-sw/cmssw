/*! \brief   Implementation of methods of TTCluster
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Nicola Pozzobon
 *  \author Emmanuele Salvati
 *  \date   2013, Jul 12
 *
 */

#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"

/// Cluster width
template <>
unsigned int TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi> >::findWidth() const {
  int rowMin = 99999999;
  int rowMax = 0;
  /// this is only the actual size in RPhi
  for (unsigned int i = 0; i < theHits.size(); i++) {
    int row = 0;
    if (this->getRows().empty()) {
      row = theHits[i]->row();
    } else {
      row = this->getRows()[i];
    }
    if (row < rowMin)
      rowMin = row;
    if (row > rowMax)
      rowMax = row;
  }
  return abs(rowMax - rowMin + 1);  /// This takes care of 1-Pixel clusters
}

/// Get hit local coordinates
template <>
MeasurementPoint TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi> >::findHitLocalCoordinates(
    unsigned int hitIdx) const {
  /// NOTE in this case, DO NOT add 0.5
  /// to get the center of the pixel
  if (this->getRows().empty() || this->getCols().empty()) {
    MeasurementPoint mp(theHits[hitIdx]->row(), theHits[hitIdx]->column());
    return mp;
  } else {
    int row = this->getRows()[hitIdx];
    int col = this->getCols()[hitIdx];
    MeasurementPoint mp(row, col);
    return mp;
  }
}

/// Unweighted average local cluster coordinates
template <>
MeasurementPoint
TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi> >::findAverageLocalCoordinates() const {
  double averageCol = 0.0;
  double averageRow = 0.0;

  /// Loop over the hits and calculate the average coordinates
  if (!theHits.empty()) {
    if (this->getRows().empty() || this->getCols().empty()) {
      typename std::vector<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi> >::const_iterator hitIter;
      for (hitIter = theHits.begin(); hitIter != theHits.end(); hitIter++) {
        averageCol += (*hitIter)->column();
        averageRow += (*hitIter)->row();
      }
      averageCol /= theHits.size();
      averageRow /= theHits.size();
    } else {
      for (unsigned int j = 0; j < theHits.size(); j++) {
        averageCol += theCols[j];
        averageRow += theRows[j];
      }
      averageCol /= theHits.size();
      averageRow /= theHits.size();
    }
  }
  return MeasurementPoint(averageRow, averageCol);
}

/// Unweighted average local cluster coordinates, using center of the strips
template <>
MeasurementPoint TTCluster<
    edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi> >::findAverageLocalCoordinatesCentered() const {
  double averageCol = 0.0;
  double averageRow = 0.0;

  /// Loop over the hits and calculate the average coordinates
  if (!theHits.empty()) {
    if (this->getRows().empty() || this->getCols().empty()) {
      typename std::vector<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi> >::const_iterator hitIter;
      for (hitIter = theHits.begin(); hitIter != theHits.end(); hitIter++) {
        averageCol += (*hitIter)->column() + 0.5;
        averageRow += (*hitIter)->row() + 0.5;
      }
      averageCol /= theHits.size();
      averageRow /= theHits.size();
    } else {
      for (unsigned int j = 0; j < theHits.size(); j++) {
        averageCol += theCols[j] + 0.5;
        averageRow += theRows[j] + 0.5;
      }
      averageCol /= theHits.size();
      averageRow /= theHits.size();
    }
  }
  return MeasurementPoint(averageRow, averageCol);
}

/// Coordinates stored locally
template <>
std::vector<int> TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi> >::findRows() const {
  std::vector<int> temp;
  temp.reserve(theHits.size());
  for (unsigned int i = 0; i < theHits.size(); i++) {
    temp.push_back(theHits[i]->row());
  }
  return temp;
}

template <>
std::vector<int> TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi> >::findCols() const {
  std::vector<int> temp;
  temp.reserve(theHits.size());
  for (unsigned int i = 0; i < theHits.size(); i++) {
    temp.push_back(theHits[i]->column());
  }
  return temp;
}
