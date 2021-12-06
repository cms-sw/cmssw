/** \Class RPCMaskReClusterizer
 *  \author J.C. Sanabria -- UniAndes, Bogota
 */

#include "RPCCluster.h"
#include "RPCClusterizer.h"
#include "RPCMaskReClusterizer.h"

RPCClusterContainer RPCMaskReClusterizer::doAction(const RPCDetId& id,
                                                   RPCClusterContainer& initClusters,
                                                   const RollMask& mask) const {
  RPCClusterContainer finClusters;
  if (initClusters.empty())
    return finClusters;

  RPCCluster prev = *initClusters.begin();
  for (auto cl = std::next(initClusters.begin()); cl != initClusters.end(); ++cl) {
    // Merge this cluster if it is adjacent by 1 masked strip
    // Note that the RPCClusterContainer collection is sorted in DECREASING ORDER of strip #
    // So the prev. cluster is placed after the current cluster (check the < operator of RPCCluster carefully)
    if ((prev.firstStrip() - cl->lastStrip()) == 2 and this->get(mask, cl->lastStrip() + 1) and prev.bx() == cl->bx()) {
      RPCCluster merged(cl->firstStrip(), prev.lastStrip(), cl->bx());
      prev = merged;
    } else {
      finClusters.insert(prev);
      prev = *cl;
    }
  }

  // Finalize by putting the last cluster to the collection
  finClusters.insert(prev);

  return finClusters;
}

bool RPCMaskReClusterizer::get(const RollMask& mask, int strip) const { return mask.test(strip - 1); }
