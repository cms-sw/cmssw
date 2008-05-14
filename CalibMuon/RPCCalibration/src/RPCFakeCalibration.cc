#include "CalibMuon/RPCCalibration/interface/RPCFakeCalibration.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibMuon/RPCCalibration/interface/RPCSimSetUp.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"

#include <cmath>
#include <math.h>
#include <iostream>
#include <memory>
#include <fstream>

RPCFakeCalibration::RPCFakeCalibration( const edm::ParameterSet& pset ) : RPCPerformanceESSource(pset) {

  edm::LogInfo("RPCFakeCalibration::RPCFakeCalibration");
  //  printdebug_ = pset.getUntrackedParameter<bool>("printDebug", false);
  std::cout<<"SONO NEL COSTRUTTORE DI RPCAFKECALIB"<<std::endl;

  theRPCSimSetUp  =  new RPCSimSetUp(pset);
  std::cout<<"SONO NEL COSTRUTTORE DI RPCAFKECALIB DOPO SIMSETUP"<<std::endl;

}

RPCStripNoises * RPCFakeCalibration::makeNoise() { 

  std::cout<<"SONO NEL MAKENOISE DI RPCAFKECALIB"<<std::endl;

  RPCStripNoises * obj = new RPCStripNoises();
  
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

  for(std::map<uint32_t, std::vector<float> >::iterator it = (theRPCSimSetUp->_mapDetIdNoise).begin();
      it != (theRPCSimSetUp->_mapDetIdNoise).end(); it++){
    
    //--------------------------------------------------------------
    //i++;
    //std::cout<<" times in the cicle: " << i<< std::endl;
    //std::cout<< " it-> first " << it->first << std::endl;
    //std::cout<< " it-> second " << ((it->second))[0] << std::endl;
    //--------------------------------------------------------------
    

    tipoprova.dpid = it->first;
    
    std::cout << "(it->second).size() == " <<  (it->second).size()<< std::endl;
    
    for(unsigned int k = 0; k < 96; ++k){
      tipoprova.noise[k] = ((it->second))[k];
      tipoprova.eff[k] = (theRPCSimSetUp->getEff(it->first))[k];
    }
    tipoprova.time =  theRPCSimSetUp->getTime(it->first);


    (obj->v_noises).push_back(tipoprova);
  }

  return obj;
}
