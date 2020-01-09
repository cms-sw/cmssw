/** \Class GEMMaskReClusterizer
 *  \author J.C. Sanabria -- UniAndes, Bogota
 */
#include "RecoLocalMuon/GEMRecHit/interface/GEMCluster.h"
#include "RecoLocalMuon/GEMRecHit/interface/GEMMaskReClusterizer.h"

GEMClusterContainer GEMMaskReClusterizer::doAction(const GEMDetId& id,
                                                   GEMClusterContainer& initClusters,
                                                   const EtaPartitionMask& mask) const {
  GEMClusterContainer finClusters;
  if (initClusters.empty())
    return finClusters;

  GEMCluster prev = *initClusters.begin();
  for (auto cl = std::next(initClusters.begin()); cl != initClusters.end(); ++cl) {
    // Merge this cluster if it is adjacent by 1 masked strip
    // Note that the GEMClusterContainer collection is sorted in DECREASING ORDER of strip #
    // So the prev. cluster is placed after the current cluster (check the < operator of GEMCluster carefully)
    if ((prev.firstStrip() - cl->lastStrip()) == 2 and this->get(mask, cl->lastStrip() + 1) and prev.bx() == cl->bx()) {
      prev = GEMCluster(cl->firstStrip(), prev.lastStrip(), cl->bx());
    } else {
      finClusters.insert(prev);
      prev = *cl;
    }
  }

  // Finalize by putting the last cluster to the collection
  finClusters.insert(prev);

  return finClusters;
}

bool GEMMaskReClusterizer::get(const EtaPartitionMask& mask, int strip) const { return mask.test(strip); }
