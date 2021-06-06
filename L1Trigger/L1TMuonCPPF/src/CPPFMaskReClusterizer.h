#ifndef L1Trigger_CPPFMaskReClusterizer_h
#define L1Trigger_CPPFMaskReClusterizer_h

/** \Class CPPFMaskReClusterizer
 *  \author R. Hadjiiska -- INRNE-BAS, Sofia
 */

#include "CPPFRPCRollMask.h"
#include "CPPFClusterContainer.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

class CPPFMaskReClusterizer {
public:
  CPPFMaskReClusterizer(){};
  ~CPPFMaskReClusterizer(){};
  CPPFClusterContainer doAction(const RPCDetId& id, CPPFClusterContainer& initClusters, const CPPFRollMask& mask) const;
  bool get(const CPPFRollMask& mask, int strip) const;
};

#endif
