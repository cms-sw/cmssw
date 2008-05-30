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
#include "CondTools/RPC/interface/RPCDBPerformanceHandler.h"
#include <iostream>
#include <fstream>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimMuon/RPCDigitizer/src/RPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigitizer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "CondTools/RPC/interface/RPCDBSimSetUp.h"

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

RPCDBPerformanceHandler::RPCDBPerformanceHandler(const edm::ParameterSet& pset) :
  m_since(pset.getUntrackedParameter<int >("firstSince")),
  dataTag(   pset.getParameter<std::string>  (  "tag" ) ){


std::cout<< " first map handling " << std::endl;

   theRPCSimSetUp  =  new RPCDBSimSetUp(pset);
  
  std::cout<< " after map handling " << std::endl;

}

RPCDBPerformanceHandler::~RPCDBPerformanceHandler(){}


void RPCDBPerformanceHandler::getNewObjects(){

  std::cout << " - > getNewObjects\n" << 
    //check whats already inside of database
    "got offlineInfo"<<
    tagInfo().name << ", size " << tagInfo().size 
	    << ", last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;

  RPCStripNoises* obj = new RPCStripNoises();

  std::map< int, std::vector<double> >::iterator itc;
  for(itc = (theRPCSimSetUp->_clsMap).begin();itc != (theRPCSimSetUp->_clsMap).end();++itc){
    std::cout<<itc->first<<"  "<<(itc->second).size()<<std::endl;

    for(unsigned int n = 0; n < (itc->second).size();++n){
      std::cout<<"CLS: "<<(itc->second)[n]<<std::endl;
      (obj->v_cls).push_back((itc->second)[n]);
    }
  }

  RPCStripNoises::NoiseItem tipoprova;

  std::cout<< " map size " << theRPCSimSetUp->_mapDetIdNoise.size() << std::endl;
  int i = 0;
  for(std::map<uint32_t, std::vector<float> >::iterator it = (theRPCSimSetUp->_mapDetIdNoise).begin(); 
      it != (theRPCSimSetUp->_mapDetIdNoise).end(); it++){

    //--------------------------------------------------------------
    //i++;
    //std::cout<<" times in the cicle: " << i<< std::endl; 
    //std::cout<< " it-> first " << it->first << std::endl;
    //std::cout<< " it-> second " << ((it->second))[0] << std::endl;
    //--------------------------------------------------------------    


    tipoprova.dpid = it->first;

    //std::cout << "(it->second).size() == " <<  (it->second).size()<< std::endl;

   for(unsigned int k = 0; k < 96; ++k){
       tipoprova.noise[k] = ((it->second))[k];
       tipoprova.eff[k] = (theRPCSimSetUp->getEff(it->first))[k];
    }
   tipoprova.time =  theRPCSimSetUp->getTime(it->first);


    (obj->v_noises).push_back(tipoprova);
    edm::LogError("RPCStripNoisesBuilder")<<"[RPCStripNoisesBuilder::analyze] detid already exists"<<std::endl;
    
  }

  // prepare for transfer:
  m_to_transfer.push_back( std::make_pair((RPCStripNoises*)obj,m_since) );
 
}

std::string RPCDBPerformanceHandler::id() const {
  return dataTag;
}





