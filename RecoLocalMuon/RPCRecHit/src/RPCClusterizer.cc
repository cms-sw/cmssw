#include "RPCClusterizer.h"
#include "RPCCluster.h"
#include "RPCClusterContainer.h"


RPCClusterizer::RPCClusterizer()
{
}

RPCClusterizer::~RPCClusterizer()
{
}
 
RPCClusterContainer
RPCClusterizer::doAction(const RPCDigiCollection::Range& digiRange){
  RPCClusterContainer cls;
  for (RPCDigiCollection::const_iterator digi = digiRange.first;
       digi != digiRange.second;
       digi++) {
    RPCCluster cl(digi->strip(),digi->strip(),digi->bx());
    cls.insert(cl);
  }
  this->doActualAction(cls);
  return cls;
}


void RPCClusterizer::doActualAction(RPCClusterContainer& initialclusters){
  
  RPCClusterContainer finalCluster;
  RPCCluster prev;

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
 

