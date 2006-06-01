#ifndef RecoLocalMuon_RPCClusterizer_h
#define RecoLocalMuon_RPCClusterizer_h
/** \class RPCClusterizer
 *  $Date: 2006/05/30 09:26:57 $
 *  $Revision: 1.3 $
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
  void doActualAction(RPCClusterContainer& initialclusters);

 private:
  RPCClusterContainer cls;
};
#endif
