#include "RPCClusterizer.h"

RPCClusterContainer RPCClusterizer::doAction(const RPCDigiCollection::Range& digiRange)
{
  RPCClusterContainer initialCluster, finalCluster;
  // Return empty container for null input
  if ( std::distance(digiRange.second, digiRange.first) == 0 ) return finalCluster;

  // Start from single digi recHits
  for ( auto digi = digiRange.first; digi != digiRange.second; ++digi ) {
    initialCluster.insert(RPCCluster(digi->strip(), digi->strip(), digi->bx()));
  }
  if ( initialCluster.empty() ) return finalCluster; // Confirm the collection is valid

  // Start from the first initial cluster
  RPCCluster prev = *initialCluster.begin();

  // Loop over the remaining digis
  // Note that the last one remains as open in this loop
  for ( auto cl = std::next(initialCluster.begin()); cl != initialCluster.end(); ++cl ) {
    if ( prev.isAdjacent(*cl) ) {
      // Merged digi to the previous one
      prev.merge(*cl);
    }
    else {
      // Close the previous cluster and start new cluster
      finalCluster.insert(prev);
      prev = *cl;
    }
  }

  // Finalize by adding the last cluster
  finalCluster.insert(prev);

  return finalCluster;
} 


