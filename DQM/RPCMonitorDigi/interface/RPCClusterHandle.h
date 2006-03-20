#ifndef RPCClusterHandle_H
#define RPCClusterHandle_H

/** \class RPCClusterHandle
 *
 * Collection of strip with signal id's and/or of digi.
 * Using either of these  collections it finds out number
 * of clusters and relative cluster sizes
 *
 * (cluster=set of adjacient strips with signal)
 *
 *  $Date: 2006/03/09 10:15:50 $
 *  $Revision: 1.1 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/RPCDigi/interface/RPCDigi.h>

#include <vector>
#include <string>

class RPCClusterHandle
{
public:
   ///Creator
   RPCClusterHandle(std::string nameForLogFile);
   ///Destructor
   ~RPCClusterHandle();
   ///Add strip Id to colletion
   void addStrip(int strip);
   ///Set containers to zero
   void reset();
   ///Get cluster size
   int  size();
   ///Get strips in the cluster 
   std::vector<int> getStripsInCluster();
   
   ///Find number of clusters and their multicplicyty
   std::vector<int> findClustersFromStrip();
   
   ///Utility methods
   bool searchPrevious(int adjacentStrip);
   bool searchNext(int adjacentStrip);
   
private:
   std::vector<int>     stripInCluster;
   
   std::string nameInLog;

};




#endif
