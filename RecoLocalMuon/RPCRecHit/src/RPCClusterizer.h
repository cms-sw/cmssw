#ifndef RecoLocalMuon_RPCClusterizer_h
#define RecoLocalMuon_RPCClusterizer_h
/** \class RPCClusterizer
 *  $Date: 2006/06/01 22:27:09 $
 *  $Revision: 1.4 $
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
