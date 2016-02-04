// -*- C++ -*-
//
// Package:    RPCTriggerBxOrConfig
// Class:      RPCTriggerBxOrConfig
// 
/**\class RPCTriggerBxOrConfig RPCTriggerBxOrConfig.h L1TriggerConfig/RPCTriggerBxOrConfig/src/RPCTriggerBxOrConfig.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Wed Apr  9 13:57:29 CEST 2008
// $Id: RPCTriggerBxOrConfig.cc,v 1.1 2010/02/26 15:50:55 fruboes Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1RPCBxOrConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCBxOrConfig.h"



//
// class decleration
//

class RPCTriggerBxOrConfig : public edm::ESProducer {
   public:
      RPCTriggerBxOrConfig(const edm::ParameterSet&);
      ~RPCTriggerBxOrConfig();

      typedef std::auto_ptr<L1RPCBxOrConfig> ReturnType;

      ReturnType produce(const L1RPCBxOrConfigRcd&);
   private:
    int m_firstBX;
    int m_lastBX;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RPCTriggerBxOrConfig::RPCTriggerBxOrConfig(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   m_firstBX = iConfig.getParameter<int>("firstBX");
   m_lastBX = iConfig.getParameter<int>("lastBX"); 


}


RPCTriggerBxOrConfig::~RPCTriggerBxOrConfig()
{
 

}


//
// member functions
//

// ------------ method called to produce the data  ------------
RPCTriggerBxOrConfig::ReturnType
RPCTriggerBxOrConfig::produce(const L1RPCBxOrConfigRcd& iRecord)
{
   using namespace edm::es;
   std::auto_ptr<L1RPCBxOrConfig> pRPCTriggerBxOrConfig = std::auto_ptr<L1RPCBxOrConfig>( new L1RPCBxOrConfig() );

   if (m_firstBX > m_lastBX )
        throw cms::Exception("BadConfig") << " firstBX < m_lastBX  " << "\n";

   pRPCTriggerBxOrConfig->setFirstBX(m_firstBX);
   pRPCTriggerBxOrConfig->setLastBX(m_lastBX);

   return pRPCTriggerBxOrConfig ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RPCTriggerBxOrConfig);
