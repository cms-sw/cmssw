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
// $Id: L1CondDBPayloadWriter.cc,v 1.18 2010/02/16 21:59:24 wsun Exp $
//
//


// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
   : m_writeL1TriggerKey( iConfig.getParameter< bool >( "writeL1TriggerKey" )),
     m_writeConfigData( iConfig.getParameter< bool >( "writeConfigData" ) ),
     m_overwriteKeys( iConfig.getParameter< bool >( "overwriteKeys" ) ),
     m_logTransactions( iConfig.getParameter<bool>( "logTransactions" ) ),
     m_newL1TriggerKeyList( iConfig.getParameter< bool >( "newL1TriggerKeyList" ) )
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
   L1TriggerKeyList oldKeyList ;

   if( !m_newL1TriggerKeyList )
     {
       if( !m_writer.fillLastTriggerKeyList( oldKeyList ) )
	 {
	   edm::LogError( "L1-O2O" )
	     << "Problem getting last L1TriggerKeyList" ;
	 }
     }

   L1TriggerKeyList* keyList = 0 ;

   // Write L1TriggerKey to ORCON with no IOV
   std::string token ;
   ESHandle< L1TriggerKey > key ;

   // Before calling writePayload(), check if TSC key is already in
   // L1TriggerKeyList.  writePayload() will not catch this situation in
   // the case of dummy configurations.
   bool triggerKeyOK = true ;
   try
     {
       // Get L1TriggerKey
       iSetup.get< L1TriggerKeyRcd >().get( key ) ;

       if( !m_overwriteKeys )
	 {
	   triggerKeyOK = oldKeyList.token( key->tscKey() ) == "" ;
	 }
     }
   catch( l1t::DataAlreadyPresentException& ex )
     {
       triggerKeyOK = false ;
       edm::LogVerbatim( "L1-O2O" ) << ex.what() ;
     }

   if( triggerKeyOK && m_writeL1TriggerKey )
     {
       edm::LogVerbatim( "L1-O2O" )
         << "Object key for L1TriggerKeyRcd@L1TriggerKey: "
         << key->tscKey() ;
       token = m_writer.writePayload( iSetup,
				      "L1TriggerKeyRcd@L1TriggerKey" ) ;
     }

   // If L1TriggerKey is invalid, then all configuration data is already in DB
   bool throwException = false ;

   if( !token.empty() || !m_writeL1TriggerKey )
   {
      // Record token in L1TriggerKeyList
      if( m_writeL1TriggerKey )
	{
	  keyList = new L1TriggerKeyList( oldKeyList ) ;
	  if( !( keyList->addKey( key->tscKey(),
				  token,
				  m_overwriteKeys ) ) )
	    {
	      throw cond::Exception( "L1CondDBPayloadWriter: TSC key "
				     + key->tscKey()
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
	      // Do nothing if object key is null.
	      if( it->second == L1TriggerKey::kNullKey )
		{
		  edm::LogVerbatim( "L1-O2O" )
		    << "L1CondDBPayloadWriter: null object key for "
		    << it->first << "; skipping this record." ;
		}
	      else
		{
		  // Check key is new before writing
		  if( oldKeyList.token( it->first, it->second ) == "" ||
		      m_overwriteKeys )
		    {
		      // Write data to ORCON with no IOV
		      if( oldKeyList.token( it->first, it->second ) != "" )
			{
			  edm::LogVerbatim( "L1-O2O" )
			    << "*** Overwriting payload: object key for "
			    << it->first << ": " << it->second ;
			}
		      else
			{
			  edm::LogVerbatim( "L1-O2O" )
			    << "object key for "
			    << it->first << ": " << it->second ;
			}

		      try
			{
			  token = m_writer.writePayload( iSetup, it->first ) ;
			}
		      catch( l1t::DataInvalidException& ex )
			{
			  edm::LogVerbatim( "L1-O2O" )
			    << ex.what()
			    << " Skipping to next record." ;

			  throwException = true ;

			  continue ;
			}

		      if( !token.empty() )
			{
			  // Record token in L1TriggerKeyList
			  if( !keyList )
			    {
			      keyList = new L1TriggerKeyList( oldKeyList ) ;
			    }

			  if( !( keyList->addKey( it->first,
						  it->second,
						  token,
						  m_overwriteKeys ) ) )
			    {
			      // This should never happen because of the check
			      // above, but just in case....
			      throw cond::Exception(
				"L1CondDBPayloadWriter: subsystem key "
				+ it->second + " for " + it->first
				+ " already in L1TriggerKeyList" ) ;
			    }
			}
		    }
		  else
		    {
		      edm::LogVerbatim( "L1-O2O" )
			<< "L1CondDBPayloadWriter: object key "
			<< it->second << " for " << it->first
			<< " already in L1TriggerKeyList" ;
		    }
		}
	    }
	}
   }

   if( keyList )
   {
      // Write L1TriggerKeyList to ORCON
      m_writer.writeKeyList( keyList, 0, m_logTransactions ) ;
   }

   if( throwException )
     {
       throw l1t::DataInvalidException( "Payload problem found." ) ;
     }
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1CondDBPayloadWriter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1CondDBPayloadWriter::endJob() {
}

//define this as a plug-in
//DEFINE_FWK_MODULE(L1CondDBPayloadWriter);
