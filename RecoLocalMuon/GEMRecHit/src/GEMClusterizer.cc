#include "RecoLocalMuon/GEMRecHit/interface/GEMClusterizer.h"

GEMClusterContainer GEMClusterizer::doAction(const GEMDigiCollection::Range& digiRange, const EtaPartitionMask& mask) {
  GEMClusterContainer initialCluster, finalCluster;
  // Return empty container for null input
  if (std::distance(digiRange.second, digiRange.first) == 0)
    return finalCluster;

  // Start from single digi recHits
  for (auto digi = digiRange.first; digi != digiRange.second; ++digi) {
    if (mask.test(digi->strip()))
      continue;
    GEMCluster cl(digi->strip(), digi->strip(), digi->bx());
    initialCluster.insert(cl);
  }
  if (initialCluster.empty())
    return finalCluster;  // Confirm the collection is valid

  // Start from the first initial cluster
  GEMCluster prev = *initialCluster.begin();

  // Loop over the remaining digis
  // Note that the last one remains as open in this loop
  for (auto cl = std::next(initialCluster.begin()); cl != initialCluster.end(); ++cl) {
    if (prev.isAdjacent(*cl)) {
      // Merged digi to the previous one
      prev.merge(*cl);
    } else {
      // Close the previous cluster and start new cluster
      finalCluster.insert(prev);
      prev = *cl;
    }
  }

  // Finalize by adding the last cluster
  finalCluster.insert(prev);

  return finalCluster;
}
