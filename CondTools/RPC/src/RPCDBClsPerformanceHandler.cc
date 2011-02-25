#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// class decleration
//
#include "CondTools/RPC/interface/RPCDBClsPerformanceHandler.h"
#include <iostream>
#include <fstream>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "CondTools/RPC/interface/RPCDBClsSimSetUp.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"

#include <cmath>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include<cstring>
#include<string>
#include<vector>
#include<stdlib.h>
#include <utility>
#include <map>

using namespace edm;

RPCDBClsPerformanceHandler::RPCDBClsPerformanceHandler(const edm::ParameterSet& pset) :
  m_since(pset.getUntrackedParameter<int >("firstSince")),
  dataTag(   pset.getParameter<std::string>  (  "tag" ) ){
  theRPCClsSimSetUp  =  new RPCDBClsSimSetUp(pset);
}

RPCDBClsPerformanceHandler::~RPCDBClsPerformanceHandler(){}


void RPCDBClsPerformanceHandler::getNewObjects(){
  std::cout << " - > getNewObjects\n" << 
    //check whats already inside of database
    "got offlineInfo"<<
    tagInfo().name << ", size " << tagInfo().size 
	    << ", last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;


  RPCClusterSize * obj = new RPCClusterSize();
  RPCClusterSize::ClusterSizeItem rpcClsItem;
  
  for(std::map<uint32_t, std::vector<double> >::iterator it 
        = (theRPCClsSimSetUp->_mapDetClsMap).begin();
      it != (theRPCClsSimSetUp->_mapDetClsMap).end(); it++){
    
    rpcClsItem.dpid =  it->first;
    
    for(unsigned int k = 0; k < 100; k++){
      
      
            rpcClsItem.clusterSize = (theRPCClsSimSetUp->getCls(it->first))[k];
            (obj->v_cls).push_back(rpcClsItem);
    }
  }
  






    
    edm::LogError("RPCClustersSizeBuilder")<<"[RPCClusterSizeBuilder::analyze] detid already exists"<<std::endl;

  m_to_transfer.push_back( std::make_pair((RPCClusterSize*)obj,m_since) );
 
}

std::string RPCDBClsPerformanceHandler::id() const {
  return dataTag;
}





