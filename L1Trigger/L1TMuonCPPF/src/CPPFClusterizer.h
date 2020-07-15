#ifndef L1Trigger_CPPFClusterizer_h
#define L1Trigger_CPPFClusterizer_h
/** \class CPPFClusterizer
 *  \author M. Maggi -- INFN Bari
 */

#include "CPPFClusterContainer.h"
#include "CPPFCluster.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

class CPPFClusterizer {
public:
  CPPFClusterizer(){};
  ~CPPFClusterizer(){};
  CPPFClusterContainer doAction(const RPCDigiCollection::Range& digiRange);
};
#endif
