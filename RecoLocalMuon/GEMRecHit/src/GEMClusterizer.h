#ifndef RecoLocalMuon_GEMRecHit_GEMClusterizer_h
#define RecoLocalMuon_GEMRecHit_GEMClusterizer_h
/** \class GEMClusterizer
 *  \author M. Maggi -- INFN Bari
 */

#include "GEMClusterContainer.h"
#include "GEMCluster.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

class GEMClusterizer{
 public:
  GEMClusterizer() {};
  ~GEMClusterizer() {};
  GEMClusterContainer doAction(const GEMDigiCollection::Range& digiRange);
};
#endif
