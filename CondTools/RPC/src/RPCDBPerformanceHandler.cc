#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

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

#include "CondTools/RPC/interface/RPCDBSimSetUp.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"

#include <cmath>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <cstdlib>
#include <utility>
#include <map>

using namespace edm;

RPCDBPerformanceHandler::RPCDBPerformanceHandler(const edm::ParameterSet& pset)
    : m_since(pset.getUntrackedParameter<int>("firstSince")), dataTag(pset.getParameter<std::string>("tag")) {
  theRPCSimSetUp = new RPCDBSimSetUp(pset);
}

RPCDBPerformanceHandler::~RPCDBPerformanceHandler() {}

void RPCDBPerformanceHandler::getNewObjects() {
  std::cout << " - > getNewObjects\n"
            <<
      //check whats already inside of database
      "got offlineInfo" << tagInfo().name << ", size " << tagInfo().size << ", last object valid since "
            << tagInfo().lastInterval.since << std::endl;

  RPCStripNoises* obj = new RPCStripNoises();

  std::map<int, std::vector<double> >::iterator itc;
  for (itc = (theRPCSimSetUp->_clsMap).begin(); itc != (theRPCSimSetUp->_clsMap).end(); ++itc) {
    for (unsigned int n = 0; n < (itc->second).size(); ++n) {
      (obj->v_cls).push_back((itc->second)[n]);
    }
  }

  RPCStripNoises::NoiseItem tipoprova;

  for (std::map<uint32_t, std::vector<float> >::iterator it = (theRPCSimSetUp->_mapDetIdNoise).begin();
       it != (theRPCSimSetUp->_mapDetIdNoise).end();
       it++) {
    tipoprova.dpid = it->first;
    tipoprova.time = theRPCSimSetUp->getTime(it->first);

    for (unsigned int k = 0; k < 96; ++k) {
      tipoprova.noise = ((it->second))[k];
      tipoprova.eff = (theRPCSimSetUp->getEff(it->first))[k];
      (obj->v_noises).push_back(tipoprova);
    }

    edm::LogError("RPCStripNoisesBuilder") << "[RPCStripNoisesBuilder::analyze] detid already exists" << std::endl;
  }

  // prepare for transfer:
  m_to_transfer.push_back(std::make_pair((RPCStripNoises*)obj, m_since));
}

std::string RPCDBPerformanceHandler::id() const { return dataTag; }
