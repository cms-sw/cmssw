/** \file
 *
 *  implementation of RPCClusterHandle class
 *
 *  $Date: 2006/02/10 10:42:57 $
 *  $Revision: 1.4 $
 *
 * \author Ilaria Segoni
 */

#include <DQM/RPCMonitorDigi/interface/RPCClusterHandle.h>

///Log messages
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
///Creator  
RPCClusterHandle::RPCClusterHandle(std::string nameForLogFile){  
	nameInLog = nameForLogFile;
}

///Dectructor
RPCClusterHandle::~RPCClusterHandle(){}

void RPCClusterHandle::addDigi(RPCDigi myDigi){
	digiInCluster.push_back(myDigi);
}

void RPCClusterHandle::addStrip(int strip){
	stripInCluster.push_back(strip);
}

void RPCClusterHandle::reset(){
	digiInCluster.clear();
} 

int  RPCClusterHandle::size(){
	return digiInCluster.size();
}

std::vector<RPCDigi> RPCClusterHandle::getDigisInCluster(){
	return digiInCluster;

}
std::vector<int> RPCClusterHandle::getStripsInCluster(){
	return stripInCluster ;
}




std::vector<int> RPCClusterHandle::findClustersFromStrip(){
 
 std::vector<int> clustersMultiplicity;
 clustersMultiplicity.clear();
 
 std::vector<int> stripAccounted;
 stripAccounted.clear();
 
 edm::LogInfo (nameInLog) <<"Beginning of findClustersFromStrip";

 std::vector<int> stripUsed;
 stripUsed.clear();
 
 for(std::vector<int>::iterator str = stripInCluster.begin(); str != stripInCluster.end(); ++str ){
	int seedStrip=*str;
 	
	//edm::LogInfo (nameInLog) <<"Seed strip= "<<seedStrip;
	///check that it has not been used already to form a cluster
	bool used = false;
	for(std::vector<int>::iterator allStrips=stripUsed.begin(); allStrips !=
 	     stripUsed.end(); ++allStrips){
		if ( ((*allStrips)-1) == seedStrip) used=true; 
	}
	
	//edm::LogInfo (nameInLog) <<"sonoqui 0 ";
	if(!used){ 
        	stripUsed.push_back(seedStrip);
		int multHigh =0;
		int seedHigh=seedStrip;
	
		while(1){
			//edm::LogInfo (nameInLog) <<"loop high strip= "<<seedHigh;
			if( this->searchNext(seedHigh) ){
				++multHigh;
				++seedHigh;
				stripUsed.push_back(seedHigh);
			}else{
				break;
			}	
		}
	
		int multLow =0;
		int seedLow=seedStrip;
		while(1){
			if( this->searchPrevious(seedLow) ){
				//edm::LogInfo (nameInLog) <<"loop low strip= "<<seedLow;
				++multLow;
		  	        ++seedLow;
				stripUsed.push_back(seedLow);
			}else{
				break;
			}   
		}
	  
		if(multLow+multHigh) clustersMultiplicity.push_back(multLow+multHigh+1);
	}	
 }
 
 return clustersMultiplicity;

}

bool RPCClusterHandle::searchPrevious(int seedStr){

///check that it is not in the middle of a cluster
 for(std::vector<int>::iterator allStrips=stripInCluster.begin(); allStrips !=
 	stripInCluster.end(); ++allStrips){
		if ( ((*allStrips)+1) == seedStr) return true; 
		
 }

 return false;
}

bool RPCClusterHandle::searchNext(int seedStr){

 for(std::vector<int>::iterator allStrips=stripInCluster.begin(); allStrips !=
 	stripInCluster.end(); ++allStrips){
		if ( ((*allStrips)-1) == seedStr) return true; 
		
 }

 return false;
}

