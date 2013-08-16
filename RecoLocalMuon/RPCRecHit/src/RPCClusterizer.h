#ifndef RecoLocalMuon_RPCClusterizer_h
#define RecoLocalMuon_RPCClusterizer_h
/** \class RPCClusterizer
 *  \author M. Maggi -- INFN Bari
 */

#include "RPCClusterContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

class RPCCluster;
class RPCClusterizer{
 public:
  RPCClusterizer();
  ~RPCClusterizer();
  RPCClusterContainer doAction(const RPCDigiCollection::Range& digiRange);

 private:
  RPCClusterContainer doActualAction(RPCClusterContainer& initialclusters);

 private:
  RPCClusterContainer cls;
};
#endif
