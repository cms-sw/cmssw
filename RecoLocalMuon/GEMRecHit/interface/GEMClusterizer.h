#ifndef RecoLocalMuon_GEMRecHit_GEMClusterizer_h
#define RecoLocalMuon_GEMRecHit_GEMClusterizer_h
/** \class GEMClusterizer
 *  \author M. Maggi -- INFN Bari
 */

#include "RecoLocalMuon/GEMRecHit/interface/GEMClusterContainer.h"
#include "RecoLocalMuon/GEMRecHit/interface/GEMCluster.h"
#include "RecoLocalMuon/GEMRecHit/interface/GEMEtaPartitionMask.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

class GEMClusterizer {
public:
  GEMClusterizer(){};
  ~GEMClusterizer(){};
  GEMClusterContainer doAction(const GEMDigiCollection::Range& digiRange, const EtaPartitionMask& mask);
};
#endif
