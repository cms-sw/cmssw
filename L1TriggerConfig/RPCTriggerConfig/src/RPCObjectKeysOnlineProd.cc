// -*- C++ -*-
//
// Package:    L1TriggerConfig
// Class:      RPCObjectKeysOnlineProd
// 
/**\class RPCObjectKeysOnlineProd RPCObjectKeysOnlineProd.h L1TriggerConfig/RPCConfigProducers/src/RPCObjectKeysOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Thu Oct  2 19:35:26 CEST 2008
// $Id: RPCObjectKeysOnlineProd.cc,v 1.2 2009/03/13 17:53:23 wsun Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class RPCObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
   public:
      RPCObjectKeysOnlineProd(const edm::ParameterSet&);
      ~RPCObjectKeysOnlineProd();

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) ;
   private:
      // ----------member data ---------------------------
  bool m_enableL1RPCConfig ;
  bool m_enableL1RPCConeDefinition ;
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
RPCObjectKeysOnlineProd::RPCObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBase( iConfig ),
    m_enableL1RPCConfig( iConfig.getParameter< bool >( "enableL1RPCConfig" ) ),
    m_enableL1RPCConeDefinition( iConfig.getParameter< bool >( "enableL1RPCConeDefinition" ) )
{}


RPCObjectKeysOnlineProd::~RPCObjectKeysOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
RPCObjectKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
  std::string rpcKey = pL1TriggerKey->subsystemKey( L1TriggerKey::kRPC ) ;

  if( !rpcKey.empty() )
    {
      if( m_enableL1RPCConfig )
	{
	  pL1TriggerKey->add( "L1RPCConfigRcd",
			      "L1RPCConfig",
			      rpcKey ) ;
	}
      if( m_enableL1RPCConeDefinition )
	{
	  pL1TriggerKey->add( "L1RPCConeDefinitionRcd",
			      "L1RPCConeDefinition",
			      rpcKey ) ;
	}
    }
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RPCObjectKeysOnlineProd);
