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
  RPCClusterContainer clsNew =this->doActualAction(cls);
  return clsNew;
}

RPCClusterContainer
RPCClusterizer::doActualAction(RPCClusterContainer& initialclusters){
  
  RPCClusterContainer finalCluster;
  RPCCluster prev;

  unsigned int j = 0;
  for(RPCClusterContainer::const_iterator i=initialclusters.begin();
      i != initialclusters.end(); i++){
    RPCCluster cl = *i;

    if(i==initialclusters.begin()){
      prev = cl;
      j++;
      if(j == initialclusters.size()){
	finalCluster.insert(prev);
      }
      else if(j < initialclusters.size()){
	continue;
      }
    }

    if(prev.isAdjacent(cl)) {
      prev.merge(cl);
      j++;
      if(j == initialclusters.size()){
	finalCluster.insert(prev);
      }
    }
    else {
      j++;
      if(j < initialclusters.size()){
	finalCluster.insert(prev);
	prev = cl;
      }
      if(j == initialclusters.size()){
	finalCluster.insert(prev);
	finalCluster.insert(cl);
      }
    }
  }

  return finalCluster;
} 
 

