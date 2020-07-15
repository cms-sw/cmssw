/** \Class CPPFMaskReClusterizer
 *  \author R. Hadjiiska -- INRNE-BAS, Sofia
 */

#include "CPPFCluster.h"
#include "CPPFClusterizer.h"
#include "CPPFMaskReClusterizer.h"

CPPFClusterContainer CPPFMaskReClusterizer::doAction(const RPCDetId& id,
                                                     CPPFClusterContainer& initClusters,
                                                     const CPPFRollMask& mask) const {
  CPPFClusterContainer finClusters;
  if (initClusters.empty())
    return finClusters;

  CPPFCluster prev = *initClusters.begin();
  for (auto cl = std::next(initClusters.begin()); cl != initClusters.end(); ++cl) {
    // Merge this cluster if it is adjacent by 1 masked strip
    // Note that the CPPFClusterContainer collection is sorted in DECREASING ORDER of strip #
    // So the prev. cluster is placed after the current cluster (check the < operator of CPPFCluster carefully)
    if ((prev.firstStrip() - cl->lastStrip()) == 2 and this->get(mask, cl->lastStrip() + 1) and prev.bx() == cl->bx()) {
      CPPFCluster merged(cl->firstStrip(), prev.lastStrip(), cl->bx());
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

bool CPPFMaskReClusterizer::get(const CPPFRollMask& mask, int strip) const { return mask.test(strip - 1); }
