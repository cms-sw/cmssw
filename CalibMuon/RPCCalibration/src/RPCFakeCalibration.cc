#include "CalibMuon/RPCCalibration/interface/RPCCalibSetUp.h"
#include "CalibMuon/RPCCalibration/interface/RPCFakeCalibration.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

RPCFakeCalibration::RPCFakeCalibration(const edm::ParameterSet &pset) : RPCPerformanceESSource(pset) {
  edm::LogInfo("RPCFakeCalibration::RPCFakeCalibration");
  theRPCCalibSetUp = new RPCCalibSetUp(pset);
}

RPCStripNoises *RPCFakeCalibration::makeNoise() {
  RPCStripNoises *obj = new RPCStripNoises();

  std::map<int, std::vector<double>>::iterator itc;
  for (itc = (theRPCCalibSetUp->_clsMap).begin(); itc != (theRPCCalibSetUp->_clsMap).end(); ++itc) {
    for (double n : (itc->second)) {
      (obj->v_cls).push_back(n);
    }
  }

  RPCStripNoises::NoiseItem tipoprova;
  for (auto &it : (theRPCCalibSetUp->_mapDetIdNoise)) {
    tipoprova.dpid = it.first;
    tipoprova.time = theRPCCalibSetUp->getTime(it.first);

    for (unsigned int k = 0; k < 96; ++k) {
      tipoprova.noise = ((it.second))[k];
      tipoprova.eff = (theRPCCalibSetUp->getEff(it.first))[k];
      (obj->v_noises).push_back(tipoprova);
    }
  }

  return obj;
}

RPCClusterSize *RPCFakeCalibration::makeCls() {
  RPCClusterSize *obj = new RPCClusterSize();
  RPCClusterSize::ClusterSizeItem rpcClsItem;

  for (auto &it : (theRPCCalibSetUp->_mapDetClsMap)) {
    rpcClsItem.dpid = it.first;

    for (unsigned int k = 0; k < 100; k++) {
      rpcClsItem.clusterSize = (theRPCCalibSetUp->getCls(it.first))[k];
      (obj->v_cls).push_back(rpcClsItem);
    }
  }
  return obj;
}
