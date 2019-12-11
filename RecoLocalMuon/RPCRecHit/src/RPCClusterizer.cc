#include "RPCClusterizer.h"
#include <iostream> // tests

RPCClusterContainer RPCClusterizer::doAction(const RPCDigiCollection::Range& digiRange)
{
  RPCClusterContainer initialCluster, finalCluster;
  // Return empty container for null input
  if ( std::distance(digiRange.second, digiRange.first) == 0 ) return finalCluster;
  

  float cl_time = 0; 
  float thrTime=1.56;  
// float speed = 19.786302;  
  // Start from single digi recHits
 
    for ( auto digi = digiRange.first; digi != digiRange.second; ++digi ) {
    RPCCluster cl(digi->strip(), digi->strip(), digi->bx());
    if ( digi->hasTime() ) cl.addTime(digi->time());
    
   
    //float timeHR = digi->time();
    //std::cout<<"timeHR="<<timeHR<<std::endl;

    if ( digi->hasY() ) {
    cl.addY(digi->coordinateY());
 //   float timeLR = timeHR-digi->coordinateY()/speed;
 //   std::cout<<"timeLR="<<timeLR<<std::endl;
}
    initialCluster.insert(cl);
    cl_time= cl.time();
    std::cout<<"cl_time="<<cl_time <<std::endl;


   }
  
  if ( initialCluster.empty() ) return finalCluster; // Confirm the collection is valid

  // Start from the first initial cluster
  RPCCluster prev = *initialCluster.begin();

  // Loop over the remaining digis
  // Note that the last one remains as open in this loop
  for ( auto cl = std::next(initialCluster.begin()); cl != initialCluster.end(); ++cl ) {
    if ( prev.isAdjacent(*cl) && abs( prev.time()- cl_time) < thrTime     ) {
      // Merged digi to the previous one
        // std::cout<<"**************prev.time()- cl.time() ="<<abs(prev.time()- cl_time)<<std::endl;
     prev.merge(*cl);
     
    }
    else {
      // Close the previous cluster and start new cluster
      finalCluster.insert(prev);
      prev = *cl;
     

    }
     

  }

  // Finalize by adding the last cluster
  finalCluster.insert(prev);

  
return finalCluster;
}

