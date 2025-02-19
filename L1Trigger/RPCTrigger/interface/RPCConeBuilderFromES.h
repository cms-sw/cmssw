#ifndef L1Trigger_RPCTrigger_RPCConeBuilderFromES_h
#define L1Trigger_RPCTrigger_RPCConeBuilderFromES_h
// -*- C++ -*-
//
// Package:     RPCTrigger
// Class  :     RPCConeBuilderFromES
// 
/**\class RPCConeBuilderFromES RPCConeBuilderFromES.h L1Trigger/RPCTrigger/interface/RPCConeBuilderFromES.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon Mar  3 13:34:15 CET 2008
// $Id: RPCConeBuilderFromES.h,v 1.6 2010/02/26 15:50:40 fruboes Exp $
//

#include "CondFormats/DataRecord/interface/L1RPCConeBuilderRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"
#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"

#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"


#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "L1Trigger/RPCTrigger/interface/RPCLogCone.h" 

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "CondFormats/L1TObjects/interface/L1RPCBxOrConfig.h"


// system include files

// user include files

// forward declarations

class RPCConeBuilderFromES
{

   public:
      RPCConeBuilderFromES();
      virtual ~RPCConeBuilderFromES();
      L1RpcLogConesVec getConesFromES(edm::Handle<RPCDigiCollection> rpcDigis, 
                                      edm::ESHandle<L1RPCConeBuilder> coneBuilder,
                                      edm::ESHandle<L1RPCConeDefinition> coneDef,
                                      edm::ESHandle<L1RPCBxOrConfig> bxOrDef,
                                      edm::ESHandle<L1RPCHwConfig> hwConfig, int bx);
      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      //RPCConeBuilderFromES(const RPCConeBuilderFromES&); // stop default

      //const RPCConeBuilderFromES& operator=(const RPCConeBuilderFromES&); // stop default

      // ---------- member data --------------------------------

};


#endif
