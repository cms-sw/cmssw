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
#include "CalibMuon/RPCCalibration/interface/RPCCalibSetUp.h"
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
  theRPCCalibSetUp  =  new RPCCalibSetUp(pset);
}

RPCStripNoises * RPCFakeCalibration::makeNoise() { 

  RPCStripNoises * obj = new RPCStripNoises();
  
  std::map< int, std::vector<double> >::iterator itc;
  for(itc = (theRPCCalibSetUp->_clsMap).begin();itc != (theRPCCalibSetUp->_clsMap).end();++itc){
    for(unsigned int n = 0; n < (itc->second).size();++n){
      (obj->v_cls).push_back((itc->second)[n]);
    }
  }
  
  RPCStripNoises::NoiseItem tipoprova;
  for(std::map<uint32_t, std::vector<float> >::iterator it = (theRPCCalibSetUp->_mapDetIdNoise).begin();
      it != (theRPCCalibSetUp->_mapDetIdNoise).end(); it++){
    
    tipoprova.dpid = it->first;
    tipoprova.time =  theRPCCalibSetUp->getTime(it->first);
    
    for(unsigned int k = 0; k < 96; ++k){
      tipoprova.noise = ((it->second))[k];
      tipoprova.eff = (theRPCCalibSetUp->getEff(it->first))[k];
      (obj->v_noises).push_back(tipoprova);
    }
  }

  return obj;
}
