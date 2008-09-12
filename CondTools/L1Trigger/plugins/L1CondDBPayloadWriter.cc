// -*- C++ -*-
//
// Package:    L1CondDBPayloadWriter
// Class:      L1CondDBPayloadWriter
// 
/**\class L1CondDBPayloadWriter L1CondDBPayloadWriter.cc CondTools/L1CondDBPayloadWriter/src/L1CondDBPayloadWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Sun Mar  2 07:05:15 CET 2008
// $Id: L1CondDBPayloadWriter.cc,v 1.2 2008/03/05 04:21:36 wsun Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/plugins/L1CondDBPayloadWriter.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

//
// class declaration
//

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1CondDBPayloadWriter::L1CondDBPayloadWriter(const edm::ParameterSet& iConfig)
   : m_writer( iConfig.getParameter< std::string >( "offlineDB" ),
	       iConfig.getParameter< std::string >( "offlineAuthentication" )),
     m_tag( iConfig.getParameter< std::string >( "L1TriggerKeyListTag" ) ),
     m_writeL1TriggerKey( iConfig.getParameter< bool >( "writeL1TriggerKey" )),
     m_writeConfigData( iConfig.getParameter< bool >( "writeConfigData" ) )
{
   //now do what ever initialization is needed

}


L1CondDBPayloadWriter::~L1CondDBPayloadWriter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1CondDBPayloadWriter::analyze(const edm::Event& iEvent,
			       const edm::EventSetup& iSetup)
{
   using namespace edm;

   // Get L1TriggerKeyList and make a copy
   ESHandle< L1TriggerKeyList > oldKeyList ;
   iSetup.get< L1TriggerKeyListRcd >().get( oldKeyList ) ;
   L1TriggerKeyList* keyList = 0 ;

   // Get L1TriggerKey
   ESHandle< L1TriggerKey > key ;
   iSetup.get< L1TriggerKeyRcd >().get( key ) ;

   // Write L1TriggerKey to ORCON with no IOV
   std::string token ;

   // Check key is new before writing
   if( m_writeL1TriggerKey &&
       oldKeyList->token( key->getTSCKey() ) == "" )
     {
       token = m_writer.writePayload( iSetup,
				      "L1TriggerKeyRcd@L1TriggerKey" ) ;
     }

   // If L1TriggerKey is invalid, then all configuration data is already in DB
   if( !token.empty() || !m_writeL1TriggerKey )
   {
      // Record token in L1TriggerKeyList
      if( m_writeL1TriggerKey )
	{
	  keyList = new L1TriggerKeyList( *oldKeyList ) ;
	  if( !( keyList->addKey( key->getTSCKey(), token ) ) )
	    {
	      throw cond::Exception( "L1CondDBPayloadWriter: TSC key "
				     + key->getTSCKey()
				     + " already in L1TriggerKeyList" ) ;
	    }
	}

      if( m_writeConfigData )
	{
	  // Loop over record@type in L1TriggerKey
	  L1TriggerKey::RecordToKey::const_iterator it =
	    key->recordToKeyMap().begin() ;
	  L1TriggerKey::RecordToKey::const_iterator end =
	    key->recordToKeyMap().end() ;

	  for( ; it != end ; ++it )
	    {
	      // Check key is new before writing
	      if( oldKeyList->token( it->first, it->second ) == "" )
		{
		  // Write data to ORCON with no IOV
		  token = m_writer.writePayload( iSetup, it->first ) ;

		  if( !token.empty() )
		    {
		      // Record token in L1TriggerKeyList
		      if( !keyList )
			{
			  keyList = new L1TriggerKeyList( *oldKeyList ) ;
			}

		      if( !( keyList->addKey( it->first, it->second,
					      token ) ) )
			{
			  throw cond::Exception(
			    "L1CondDBPayloadWriter: subsystem key "
			    + it->second + " for " + it->first
			    + " already in L1TriggerKeyList" ) ;
			}
		    }
		}
	    }
	}
   }

   if( keyList )
   {
      // Write L1TriggerKeyList to ORCON with IOV since-time = previous run
      m_writer.writeKeyList( keyList,
			     m_tag ) ;
      //			     iEvent.id().run() ) ; // since time
   }
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1CondDBPayloadWriter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1CondDBPayloadWriter::endJob() {
}

//define this as a plug-in
//DEFINE_FWK_MODULE(L1CondDBPayloadWriter);
