#ifndef RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterMergerAlgorithm_h
#define RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterMergerAlgorithm_h

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"

#include <algorithm>
#include <vector>

class Phase2TrackerClusterMergerAlgorithm {
public:
  inline void mergeDetUnit(const edmNew::DetSet<Phase2TrackerCluster1D>&,
                           Phase2TrackerCluster1DCollectionNew::FastFiller&) const;
};

void Phase2TrackerClusterMergerAlgorithm::mergeDetUnit(
    const edmNew::DetSet<Phase2TrackerCluster1D>& input,
    Phase2TrackerCluster1DCollectionNew::FastFiller& merged) const {
  if (input.empty())
    return;

  // The clusterizer emits its clusters ordered by (column, first row), but an upstream
  // producer applying a hardware cluster size cap need not, so order them explicitly.
  // Phase2TrackerCluster1D's own operator< compares the first strip while ignoring the
  // column, hence the comparator here.
  std::vector<Phase2TrackerCluster1D> ordered(input.begin(), input.end());
  std::sort(ordered.begin(), ordered.end(), [](const auto& one, const auto& other) {
    if (one.column() != other.column())
      return one.column() < other.column();
    return one.firstRow() < other.firstRow();
  });

  // Keeping the first constituent's digi rather than rebuilding one from (row, column)
  // preserves its over threshold bit, as the clusterizer itself does.
  auto ci = ordered.begin();
  Phase2TrackerDigi firstDigi = ci->firstDigi();
  unsigned int sizeCluster = ci->size();
  unsigned int HIPbit = ci->threshold();
  ++ci;
  for (; ci != ordered.end(); ++ci) {
    // Adjacent means the next cluster starts on the row right after this one ends. The
    // column has to match: the last row of a column is not adjacent to the first row of
    // the next one.
    if (ci->column() == firstDigi.column() and ci->firstRow() == firstDigi.row() + sizeCluster) {
      sizeCluster += ci->size();
      HIPbit |= ci->threshold();
    } else {
      merged.push_back(Phase2TrackerCluster1D(firstDigi, sizeCluster, HIPbit));
      firstDigi = ci->firstDigi();
      sizeCluster = ci->size();
      HIPbit = ci->threshold();
    }
  }
  merged.push_back(Phase2TrackerCluster1D(firstDigi, sizeCluster, HIPbit));
}

#endif
