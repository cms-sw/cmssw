#include "RPCClusterizer.h"
#include "RPCCluster.h"
#include "RPCClusterContainer.h"


RPCClusterizer::RPCClusterizer()
{
}

RPCClusterizer::~RPCClusterizer()
{
}
 
void RPCClusterizer::doAction(RPCClusterContainer& initialclusters){
  
  RPCClusterContainer finalCluster;
  RPCCluster prev;

  RPCDigiCollection RPCDigi = initialclusters->digi.strip();

  for(RPCClusterContainer::const_iterator i=initialclusters.begin();
      i != initialclusters.end(); i++){
    RPCCluster cl = *i;
    if(prev.isAdjacent(cl)) {
      prev.merge(cl);
    }
    else{
      finalCluster.insert(prev);
      prev == cl;
      
    }
  }
} 
 
