#ifndef L1Trigger_RPCConeBuilder_RPCConeBuilder_h
#define L1Trigger_RPCConeBuilder_RPCConeBuilder_h
// -*- C++ -*-
//
// Package:     RPCConeBuilder
// Class  :     RPCConeBuilder
// 
/**\class RPCConeBuilder RPCConeBuilder.h L1Trigger/RPCTrigger/interface/RPCConeBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon Feb 25 12:06:44 CET 2008
// $Id: RPCConeBuilder.h,v 1.2 2009/09/15 13:49:41 fruboes Exp $
//
#include <memory>
#include "boost/shared_ptr.hpp"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"
#include "CondFormats/DataRecord/interface/L1RPCConeBuilderRcd.h"


#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"

#include <map>
//#include "L1TriggerConfig/RPCConeBuilder/interface/RPCStripsRing.h"
#include "L1Trigger/RPCTrigger/interface/RPCStripsRing.h"

#include "CondFormats/DataRecord/interface/L1RPCConeDefinitionRcd.h"
class RPCConeBuilder : public edm::ESProducer {
   public:
      
      RPCConeBuilder(const edm::ParameterSet&);
      ~RPCConeBuilder() {};

      typedef boost::shared_ptr<L1RPCConeBuilder> ReturnType;
      

      ReturnType produce(const L1RPCConeBuilderRcd&);
      //ReturnType produce(const L1RPCConfigRcd&);
      void geometryCallback( const MuonGeometryRecord &);
      void coneDefCallback( const L1RPCConeDefinitionRcd &);
      
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
      
      edm::ESHandle<L1RPCConeDefinition> m_L1RPCConeDefinition;
      edm::ESHandle<RPCGeometry> m_rpcGeometry;          
      bool m_runOnceBuildCones; 
          
      RPCStripsRing::TIdToRindMap m_ringsMap;
      
};



#endif
