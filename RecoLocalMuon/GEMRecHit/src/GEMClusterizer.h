#ifndef RecoLocalMuon_GEMClusterizer_h
#define RecoLocalMuon_GEMClusterizer_h
/** \class GEMClusterizer
 *  $Date: 2013/04/24 17:16:34 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */

#include "GEMClusterContainer.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

class GEMCluster;
class GEMClusterizer{
 public:
  GEMClusterizer();
  ~GEMClusterizer();
  GEMClusterContainer doAction(const GEMDigiCollection::Range& digiRange);

 private:
  GEMClusterContainer doActualAction(GEMClusterContainer& initialclusters);

 private:
  GEMClusterContainer cls;
};
#endif
