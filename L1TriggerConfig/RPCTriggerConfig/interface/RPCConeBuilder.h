#ifndef L1TriggerConfig_RPCConeBuilder_RPCConeBuilder_h
#define L1TriggerConfig_RPCConeBuilder_RPCConeBuilder_h
// -*- C++ -*-
//
// Package:     RPCConeBuilder
// Class  :     RPCConeBuilder
// 
/**\class RPCConeBuilder RPCConeBuilder.h L1TriggerConfig/RPCConeBuilder/interface/RPCConeBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon Feb 25 12:06:44 CET 2008
// $Id: RPCConeBuilder.h,v 1.1 2008/03/03 14:30:09 fruboes Exp $
//
#include <memory>
#include "boost/shared_ptr.hpp"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"
#include "CondFormats/DataRecord/interface/L1RPCConeBuilderRcd.h"



#include <map>
//#include "L1TriggerConfig/RPCConeBuilder/interface/RPCStripsRing.h"
#include "L1TriggerConfig/RPCTriggerConfig/interface/RPCStripsRing.h"


class RPCConeBuilder : public edm::ESProducer {
   public:
      
      RPCConeBuilder(const edm::ParameterSet&);
      ~RPCConeBuilder() {};

      typedef boost::shared_ptr<L1RPCConeBuilder> ReturnType;
      

      ReturnType produce(const L1RPCConeBuilderRcd&);
      //ReturnType produce(const L1RPCConfigRcd&);
      void geometryCallback( const MuonGeometryRecord &);
      
   private:
     
      void buildCones(const edm::ESHandle<RPCGeometry> & rpcGeom);
      void buildConnections();
      
      std::pair<int, int> areConnected(RPCStripsRing::TIdToRindMap::iterator ref,
                        RPCStripsRing::TIdToRindMap::iterator other); /// Returns logplane number for this connection, if not connected returns -1. In second lpSize
      
      
      // ----------member data ---------------------------
      int m_towerBeg;
      int m_towerEnd;
      int m_rollBeg;
      int m_rollEnd;
      int m_hwPlaneBeg;
      int m_hwPlaneEnd;
      
      L1RPCConeBuilder::TLPSizesInTowers m_LPSizesInTowers;
      L1RPCConeBuilder::TRingsToTowers m_RingsToTowers;
      L1RPCConeBuilder::TRingsToLP m_RingsToLP;
      
      RPCStripsRing::TIdToRindMap m_ringsMap;
      
};



#endif
