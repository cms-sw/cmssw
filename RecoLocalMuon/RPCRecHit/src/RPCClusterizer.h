#ifndef RecoLocalMuon_RPCClusterizer_h
#define RecoLocalMuon_RPCClusterizer_h
/** \class RPCClusterizer
 *  $Date: 2006/04/18 16:28:31 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */

#include "RPCClusterContainer.h"
class RPCCluster;
class RPCDigi;
class RPCClusterizer{
 public:
  RPCClusterizer();
  ~RPCClusterizer();
  void doAction(RPCClusterContainer& initialclusters);
 private:
  RPCClusterContainer cls;
};
#endif
