// -*- C++ -*-
//
// Package:    RPCConeBuilder
// Class:      RPCConeBuilder
//
/**\class RPCConeBuilder RPCConeBuilder.h L1Trigger/RPCTriggerConfig/src/RPCConeBuilder.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Fri Feb 22 13:57:06 CET 2008
//
//

#include "L1Trigger/RPCTrigger/interface/RPCConeBuilder.h"

#include "CondFormats/DataRecord/interface/L1RPCConeDefinitionRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "L1Trigger/RPCTrigger/interface/RPCStripsRing.h"

#include <cmath>
#include <vector>

RPCConeBuilder::RPCConeBuilder(const edm::ParameterSet& iConfig) :
   m_towerBeg(iConfig.getParameter<int>("towerBeg")),
   m_towerEnd(iConfig.getParameter<int>("towerEnd"))
{
   setWhatProduced(this);
}

// ------------ method called to produce the data  ------------
RPCConeBuilder::ReturnType
RPCConeBuilder::produce(const L1RPCConeBuilderRcd& iRecord)
{
   auto pL1RPCConeBuilder = std::make_unique<L1RPCConeBuilder>();

   pL1RPCConeBuilder->setFirstTower(m_towerBeg);
   pL1RPCConeBuilder->setLastTower(m_towerEnd);

   edm::ESHandle<RPCGeometry> rpcGeometry;
   iRecord.getRecord<MuonGeometryRecord>().get(rpcGeometry);

   edm::ESHandle<L1RPCConeDefinition> l1RPCConeDefinition;
   iRecord.getRecord<L1RPCConeDefinitionRcd>().get(l1RPCConeDefinition);

   RPCStripsRing::TIdToRindMap ringsMap;

   buildCones(rpcGeometry.product(), l1RPCConeDefinition.product(), ringsMap);

   // Compress all connections. Since members of this class are shared
   // pointers this call will compress all data
   ringsMap.begin()->second.compressConnections();

   pL1RPCConeBuilder->setConeConnectionMap(ringsMap.begin()->second.getConnectionsMap());

   pL1RPCConeBuilder->setCompressedConeConnectionMap(
      ringsMap.begin()->second.getCompressedConnectionsMap());

   return pL1RPCConeBuilder;
}

void RPCConeBuilder::buildCones(RPCGeometry const* rpcGeom,
                                L1RPCConeDefinition const* l1RPCConeDefinition,
                                RPCStripsRing::TIdToRindMap& ringsMap) {

   // fetch geometrical data
   auto uncompressedCons = std::make_shared<L1RPCConeBuilder::TConMap>();

   int rolls = 0;
   for (auto const& it : rpcGeom->dets()) {

      if ( dynamic_cast< RPCRoll const * >( it ) == nullptr ) {
         continue;
      }

      ++rolls;
      RPCRoll const* roll = dynamic_cast< RPCRoll const*>( it );

      int ringId = RPCStripsRing::getRingId(roll);
      if ( ringsMap.find(ringId) == ringsMap.end() ) {
         ringsMap[ringId] = RPCStripsRing(roll, uncompressedCons);
      } else {
         ringsMap[ringId].addRoll(roll);
      }
   }

   // filtermixed strips, fill gaps with virtual strips
   for (auto& it : ringsMap) {

      it.second.filterOverlapingChambers();
      it.second.fillWithVirtualStrips();
   }

   // Xcheck, if rings are symettrical
   for (auto& it : ringsMap) {
      int key = it.first;
      int sign = key/100 - (key/1000)*10;

      if (sign == 0) {
         key += 100;
      } else {
         key -= 100;
      }

      if (key != 2000){// Hey 2100 has no counterring
         if (it.second.size() !=  ringsMap[key].size())
         {
            throw cms::Exception("RPCInternal") << " Size differs for ring " << key << " +- 100 \n";
         }
      }
   }
   buildConnections(l1RPCConeDefinition, ringsMap);
}

void RPCConeBuilder::buildConnections(L1RPCConeDefinition const* l1RPCConeDefinition,
                                      RPCStripsRing::TIdToRindMap& ringsMap) {

   RPCStripsRing::TIdToRindMap::iterator itRef = ringsMap.begin();
   for (;itRef != ringsMap.end(); ++itRef){ // iterate over reference rings

      RPCStripsRing::TOtherConnStructVec ringsToConnect;

      if (!itRef->second.isReferenceRing()) continue; // iterate over reference rings

      RPCStripsRing::TIdToRindMap::iterator itOther = ringsMap.begin();
      for (;itOther != ringsMap.end(); ++itOther){ // iterate over nonreference rings

         if (itOther->second.isReferenceRing()) continue; // iterate over nonreference rings

         std::pair<int,int> pr = areConnected(itRef, itOther, l1RPCConeDefinition);
         if ( pr.first != -1 ) {
            RPCStripsRing::TOtherConnStruct newOtherConn;
            newOtherConn.m_it = itOther;
            newOtherConn.m_logplane = pr.first;
            newOtherConn.m_logplaneSize = pr.second;
            ringsToConnect.push_back(newOtherConn);
         }
      } // OtherRings iteration ends

      std::pair<int,int> prRef = areConnected(itRef, itRef, l1RPCConeDefinition);
      if (prRef.first == -1){
         throw cms::Exception("RPCConfig") << " Cannot determine logplane for reference ring "
            << itRef->first << "\n ";
      }

      itRef->second.createRefConnections(ringsToConnect, prRef.first, prRef.second);

   } // RefRings iteration ends
}

// first - logplane
// second - logplanesize
std::pair<int, int>
RPCConeBuilder::areConnected(RPCStripsRing::TIdToRindMap::iterator ref,
                             RPCStripsRing::TIdToRindMap::iterator other,
                             L1RPCConeDefinition const* l1RPCConeDefinition) {

   int logplane = -1;

   // Do not connect  rolls lying on the oposite side of detector
   if ( ref->second.getEtaPartition()*other->second.getEtaPartition()<0  )
      return std::make_pair(-1,0);

   int refTowerCnt = 0;
   int index = -1;
   int refTower = -1;

   for (auto const& itRef : l1RPCConeDefinition->getRingToTowerVec()) {

      if ( itRef.m_etaPart != std::abs(ref->second.getEtaPartition())
           || itRef.m_hwPlane != std::abs(ref->second.getHwPlane()-1) // -1?
         ) {
         continue;
      }

      ++refTowerCnt;
      refTower = itRef.m_tower;

      for (auto const& itOther : l1RPCConeDefinition->getRingToTowerVec()) {

         if ( itOther.m_etaPart != std::abs(other->second.getEtaPartition())
              || itOther.m_hwPlane != std::abs(other->second.getHwPlane()-1) // -1?
              ) {
            continue;
         }

         if (itOther.m_tower == refTower) {
            index = itOther.m_index;
         }
      }
   }

   if (refTowerCnt > 1) {
      throw cms::Exception("RPCConeBuilder") << " Reference(?) ring "
         << ref->first << " "
         << "wants to be connected to " << refTowerCnt << " towers \n";
   }

   if (refTowerCnt == 0) {
      throw cms::Exception("RPCConeBuilder") << " Reference(?) ring "
         << ref->first << " "
         << " is not connected anywhere \n";
   }

   int lpSize = 0;
   if (index != -1) {

      for (auto const& it : l1RPCConeDefinition->getRingToLPVec()) {
         if (it.m_etaPart != std::abs(other->second.getEtaPartition())
             || it.m_hwPlane != std::abs(other->second.getHwPlane()-1)
             || it.m_index != index) {
            continue;
         }
         logplane = it.m_LP;
      }

      for (auto const& it : l1RPCConeDefinition->getLPSizeVec()) {
         if (it.m_tower != std::abs(refTower) || it.m_LP != logplane-1) {
            continue;
         }
         lpSize = it.m_size;
      }

      //FIXME
      if (lpSize==-1) {
         //throw cms::Exception("getLogStrip") << " lpSize==-1\n";
      }
   }
   return std::make_pair(logplane, lpSize);
}
