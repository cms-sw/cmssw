#ifndef RecoLocalMuon_RPCClusterizer_h
#define RecoLocalMuon_RPCClusterizer_h
/** \class RPCClusterizer
 *  $Date: 2006/05/07 14:09:57 $
 *  $Revision: 1.2 $
 *  \author M. Maggi -- INFN Bari
 */

#include "RPCClusterContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

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
