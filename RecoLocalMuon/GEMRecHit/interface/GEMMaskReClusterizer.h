#ifndef RecoLocalMuon_GEMRecHit_GEMMaskReClusterizer_h
#define RecoLocalMuon_GEMRecHit_GEMMaskReClusterizer_h
/** \Class GEMMaskReClusterizer
 *  \author J.C. Sanabria -- UniAndes, Bogota
 */

#include "RecoLocalMuon/GEMRecHit/interface/GEMEtaPartitionMask.h"
#include "RecoLocalMuon/GEMRecHit/interface/GEMClusterContainer.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

class GEMMaskReClusterizer {
public:
  GEMMaskReClusterizer(){};
  ~GEMMaskReClusterizer(){};
  GEMClusterContainer doAction(const GEMDetId& id,
                               GEMClusterContainer& initClusters,
                               const EtaPartitionMask& mask) const;
  bool get(const EtaPartitionMask& mask, int strip) const;
};

#endif
