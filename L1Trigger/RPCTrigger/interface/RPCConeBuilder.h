#ifndef L1Trigger_RPCTrigger_RPCConeBuilder_h
#define L1Trigger_RPCTrigger_RPCConeBuilder_h
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
//
#include <map>
#include <memory>
#include <utility>

#include "CondFormats/DataRecord/interface/L1RPCConeBuilderRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"
#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "L1Trigger/RPCTrigger/interface/RPCStripsRing.h"

class RPCConeBuilder : public edm::ESProducer {
   public:

      RPCConeBuilder(const edm::ParameterSet&);

      using ReturnType = std::unique_ptr<L1RPCConeBuilder>;

      ReturnType produce(const L1RPCConeBuilderRcd&);

   private:

      void buildCones(RPCGeometry const*,
                      L1RPCConeDefinition const*,
                      RPCStripsRing::TIdToRindMap&);

      void buildConnections(L1RPCConeDefinition const*,
                            RPCStripsRing::TIdToRindMap&);

      /// In the pair that is returned, the first element is the logplane number
      /// for this connection (if not connected returns -1) and the second element
      /// is lpSize.
      std::pair<int, int> areConnected(RPCStripsRing::TIdToRindMap::iterator ref,
                                       RPCStripsRing::TIdToRindMap::iterator other,
                                       L1RPCConeDefinition const*);

      // ----------member data ---------------------------
      int m_towerBeg;
      int m_towerEnd;
};
#endif
