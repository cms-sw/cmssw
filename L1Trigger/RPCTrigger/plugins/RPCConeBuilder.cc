// -*- C++ -*-
//
// Package:    RPCConeBuilder
// Class:      RPCConeBuilder
//
/**\class RPCConeBuilder RPCConeBuilder.h L1Trigger/RPCTriggerConfig/src/RPCConeBuilder.cc

 Description: The RPCConeBuilder class is the emulator of the Run 1 RPC PAC Trigger. 
              It is not used in the L1 Trigger decision since 2016.
	      It might be needed just for the re-emulation of the Run 1 data.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Fri Feb 22 13:57:06 CET 2008
//
//

#include "CondFormats/DataRecord/interface/L1RPCConeBuilderRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"
#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"
#include "CondFormats/DataRecord/interface/L1RPCConeDefinitionRcd.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "L1Trigger/RPCTrigger/interface/RPCStripsRing.h"

#include <cmath>
#include <vector>
#include <map>
#include <memory>
#include <utility>

class RPCConeBuilder : public edm::ESProducer {
public:
  RPCConeBuilder(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<L1RPCConeBuilder>;

  ReturnType produce(const L1RPCConeBuilderRcd&);

private:
  void buildCones(RPCGeometry const*, L1RPCConeDefinition const*, RPCStripsRing::TIdToRindMap&);

  void buildConnections(L1RPCConeDefinition const*, RPCStripsRing::TIdToRindMap&);

  /// In the pair that is returned, the first element is the logplane number
  /// for this connection (if not connected returns -1) and the second element
  /// is lpSize.
  std::pair<int, int> areConnected(RPCStripsRing::TIdToRindMap::iterator ref,
                                   RPCStripsRing::TIdToRindMap::iterator other,
                                   L1RPCConeDefinition const*);

  // ----------member data ---------------------------
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> m_rpcGeometryToken;
  edm::ESGetToken<L1RPCConeDefinition, L1RPCConeDefinitionRcd> m_l1RPCConeDefinitionToken;
  int m_towerBeg;
  int m_towerEnd;
};

RPCConeBuilder::RPCConeBuilder(const edm::ParameterSet& iConfig)
    : m_towerBeg(iConfig.getParameter<int>("towerBeg")), m_towerEnd(iConfig.getParameter<int>("towerEnd")) {
  auto cc = setWhatProduced(this);
  m_rpcGeometryToken = cc.consumes();
  m_l1RPCConeDefinitionToken = cc.consumes();
}

// ------------ method called to produce the data  ------------
RPCConeBuilder::ReturnType RPCConeBuilder::produce(const L1RPCConeBuilderRcd& iRecord) {
  auto pL1RPCConeBuilder = std::make_unique<L1RPCConeBuilder>();

  pL1RPCConeBuilder->setFirstTower(m_towerBeg);
  pL1RPCConeBuilder->setLastTower(m_towerEnd);

  RPCStripsRing::TIdToRindMap ringsMap;

  buildCones(&iRecord.get(m_rpcGeometryToken), &iRecord.get(m_l1RPCConeDefinitionToken), ringsMap);

  // Compress all connections. Since members of this class are shared
  // pointers this call will compress all data
  ringsMap.begin()->second.compressConnections();

  pL1RPCConeBuilder->setConeConnectionMap(ringsMap.begin()->second.getConnectionsMap());

  pL1RPCConeBuilder->setCompressedConeConnectionMap(ringsMap.begin()->second.getCompressedConnectionsMap());

  return pL1RPCConeBuilder;
}

void RPCConeBuilder::buildCones(RPCGeometry const* rpcGeom,
                                L1RPCConeDefinition const* l1RPCConeDefinition,
                                RPCStripsRing::TIdToRindMap& ringsMap) {
  // fetch geometrical data
  auto uncompressedCons = std::make_shared<L1RPCConeBuilder::TConMap>();

  for (auto const& it : rpcGeom->dets()) {
    RPCRoll const* roll = dynamic_cast<RPCRoll const*>(it);
    if (roll == nullptr) {
      continue;
    }

    int ringId = RPCStripsRing::getRingId(roll);
    auto found = ringsMap.find(ringId);
    if (found == ringsMap.end()) {
      ringsMap[ringId] = RPCStripsRing(roll, uncompressedCons);
    } else {
      found->second.addRoll(roll);
    }
  }

  // filtermixed strips, fill gaps with virtual strips
  for (auto& it : ringsMap) {
    it.second.filterOverlapingChambers();
    it.second.fillWithVirtualStrips();
  }

  // Xcheck, if rings are symmetrical
  for (auto& it : ringsMap) {
    int key = it.first;
    int sign = key / 100 - (key / 1000) * 10;

    if (sign == 0) {
      key += 100;
    } else {
      key -= 100;
    }

    // Check if the geometry has a complete ring:
    // note that in the case of demo chambers, the ring is not filled because only 2 sectors are added.
    // (3014 and 4014 lack counter-rings)
    if (key != 2000 && key != 3014 && key != 4014) {  // Key 2100 has no counter-ring
      if (it.second.size() != ringsMap[key].size()) {
        throw cms::Exception("RPCInternal") << " Size differs for ring " << key << " +- 100 \n";
      }
    }
  }
  buildConnections(l1RPCConeDefinition, ringsMap);
}

void RPCConeBuilder::buildConnections(L1RPCConeDefinition const* l1RPCConeDefinition,
                                      RPCStripsRing::TIdToRindMap& ringsMap) {
  RPCStripsRing::TIdToRindMap::iterator itRef = ringsMap.begin();
  for (; itRef != ringsMap.end(); ++itRef) {  // iterate over reference rings

    RPCStripsRing::TOtherConnStructVec ringsToConnect;

    if (!itRef->second.isReferenceRing())
      continue;  // iterate over reference rings

    RPCStripsRing::TIdToRindMap::iterator itOther = ringsMap.begin();
    for (; itOther != ringsMap.end(); ++itOther) {  // iterate over nonreference rings

      if (itOther->second.isReferenceRing())
        continue;  // iterate over nonreference rings

      std::pair<int, int> pr = areConnected(itRef, itOther, l1RPCConeDefinition);
      if (pr.first != -1) {
        RPCStripsRing::TOtherConnStruct newOtherConn;
        newOtherConn.m_it = itOther;
        newOtherConn.m_logplane = pr.first;
        newOtherConn.m_logplaneSize = pr.second;
        ringsToConnect.push_back(newOtherConn);
      }
    }  // OtherRings iteration ends

    std::pair<int, int> prRef = areConnected(itRef, itRef, l1RPCConeDefinition);
    if (prRef.first == -1) {
      throw cms::Exception("RPCConfig") << " Cannot determine logplane for reference ring " << itRef->first << "\n ";
    }

    itRef->second.createRefConnections(ringsToConnect, prRef.first, prRef.second);

  }  // RefRings iteration ends
}

// first - logplane
// second - logplanesize
std::pair<int, int> RPCConeBuilder::areConnected(RPCStripsRing::TIdToRindMap::iterator ref,
                                                 RPCStripsRing::TIdToRindMap::iterator other,
                                                 L1RPCConeDefinition const* l1RPCConeDefinition) {
  int logplane = -1;

  // Do not connect  rolls lying on the oposite side of detector
  if (ref->second.getEtaPartition() * other->second.getEtaPartition() < 0)
    return std::make_pair(-1, 0);

  int refTowerCnt = 0;
  int index = -1;
  int refTower = -1;

  for (auto const& itRef : l1RPCConeDefinition->getRingToTowerVec()) {
    if (itRef.m_etaPart != std::abs(ref->second.getEtaPartition()) ||
        itRef.m_hwPlane != std::abs(ref->second.getHwPlane() - 1)  // -1?
    ) {
      continue;
    }

    ++refTowerCnt;
    refTower = itRef.m_tower;

    for (auto const& itOther : l1RPCConeDefinition->getRingToTowerVec()) {
      if (itOther.m_etaPart != std::abs(other->second.getEtaPartition()) ||
          itOther.m_hwPlane != std::abs(other->second.getHwPlane() - 1)  // -1?
      ) {
        continue;
      }

      if (itOther.m_tower == refTower) {
        index = itOther.m_index;
      }
    }
  }

  if (refTowerCnt > 1) {
    throw cms::Exception("RPCConeBuilder") << " Reference(?) ring " << ref->first << " "
                                           << "wants to be connected to " << refTowerCnt << " towers \n";
  }

  if (refTowerCnt == 0) {
    throw cms::Exception("RPCConeBuilder") << " Reference(?) ring " << ref->first << " "
                                           << " is not connected anywhere \n";
  }

  int lpSize = 0;
  if (index != -1) {
    for (auto const& it : l1RPCConeDefinition->getRingToLPVec()) {
      if (it.m_etaPart != std::abs(other->second.getEtaPartition()) ||
          it.m_hwPlane != std::abs(other->second.getHwPlane() - 1) || it.m_index != index) {
        continue;
      }
      logplane = it.m_LP;
    }

    for (auto const& it : l1RPCConeDefinition->getLPSizeVec()) {
      if (it.m_tower != std::abs(refTower) || it.m_LP != logplane - 1) {
        continue;
      }
      lpSize = it.m_size;
    }

    //FIXME
    if (lpSize == -1) {
      //throw cms::Exception("getLogStrip") << " lpSize==-1\n";
    }
  }
  return std::make_pair(logplane, lpSize);
}

DEFINE_FWK_EVENTSETUP_MODULE(RPCConeBuilder);
