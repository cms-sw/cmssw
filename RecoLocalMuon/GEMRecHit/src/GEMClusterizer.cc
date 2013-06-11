#include "GEMClusterizer.h"
#include "GEMCluster.h"
#include "GEMClusterContainer.h"


GEMClusterizer::GEMClusterizer()
{
}

GEMClusterizer::~GEMClusterizer()
{
}
 
GEMClusterContainer
GEMClusterizer::doAction(const GEMDigiCollection::Range& digiRange){
  GEMClusterContainer cls;
  for (GEMDigiCollection::const_iterator digi = digiRange.first;
       digi != digiRange.second;
       digi++) {
    GEMCluster cl(digi->strip(),digi->strip(),digi->bx());
    cls.insert(cl);
  }
  GEMClusterContainer clsNew =this->doActualAction(cls);
  return clsNew;
}

GEMClusterContainer
GEMClusterizer::doActualAction(GEMClusterContainer& initialclusters){
  
  GEMClusterContainer finalCluster;
  GEMCluster prev;

  unsigned int j = 0;
  for(GEMClusterContainer::const_iterator i=initialclusters.begin();
      i != initialclusters.end(); i++){
    GEMCluster cl = *i;

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
 

