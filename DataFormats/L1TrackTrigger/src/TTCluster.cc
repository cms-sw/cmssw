
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"

/// Cluster width
template <>
unsigned int TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>::findWidth() const {
  int rowMin = 99999999;
  int rowMax = 0;
  /// this is only the actual size in RPhi
  for (int row : theRows) {
    if (row < rowMin)
      rowMin = row;
    if (row > rowMax)
      rowMax = row;
  }
  return abs(rowMax - rowMin + 1);  /// This takes care of 1-Pixel clusters
}

/// First row in cluster
template <>
unsigned int TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>::findFirstRow() const {
  int rowMin = 99999999;
  for (int row : theRows) {
    if (row < rowMin)
      rowMin = row;
  }
  return rowMin;
}

/// Get individual hit (i.e. digi) local coordinates in units of pitch
template <>
MeasurementPoint TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>::findHitLocalCoordinates(
    unsigned int hitIdx) const {
  /// NOTE in this case, DO NOT add 0.5
  /// to get the center of the pixel
  assert(hitIdx < this->getNumHits());
  int row = theRows[hitIdx];
  int col = theCols[hitIdx];
  return MeasurementPoint(row, col);
}

/// Unweighted average local cluster coordinates in units of pitch
template <>
MeasurementPoint
TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>::findAverageLocalCoordinates() const {
  const unsigned int numHits = this->getNumHits();
  double averageCol = 0.0;
  double averageRow = 0.0;

  /// Loop over the hits and calculate the average coordinates
  for (unsigned int j = 0; j < numHits; j++) {
    averageCol += theCols[j];
    averageRow += theRows[j];
  }
  averageCol /= numHits;
  averageRow /= numHits;
  return MeasurementPoint(averageRow, averageCol);
}

/// Unweighted average local cluster coordinates in units of pitch,
/// offset by 0.5*pitch to centre of pixels/strips.
template <>
MeasurementPoint TTCluster<
    edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>::findAverageLocalCoordinatesCentered() const {
  MeasurementPoint mp = this->findAverageLocalCoordinates();
  // Offset by 0.5*pitch
  MeasurementPoint mp_shift(mp.x() + 0.5, mp.y() + 0.5);
  return mp_shift;
}

/// Store coordinates locally -- Old code
template <>
void TTCluster<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>::setRowsCols(
    const std::vector<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi>>& aHits) {
  const unsigned int numHits = aHits.size();
  theRows.reserve(numHits);
  theCols.reserve(numHits);

  for (unsigned int i = 0; i < numHits; i++) {
    theRows.push_back(aHits[i]->row());
    theCols.push_back(aHits[i]->column());
  }
}
