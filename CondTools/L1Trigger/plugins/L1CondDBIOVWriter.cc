// -*- C++ -*-
//
// Package:    L1CondDBIOVWriter
// Class:      L1CondDBIOVWriter
// 
/**\class L1CondDBIOVWriter L1CondDBIOVWriter.cc CondTools/L1CondDBIOVWriter/src/L1CondDBIOVWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Sun Mar  2 20:09:46 CET 2008
// $Id: L1CondDBIOVWriter.cc,v 1.2 2008/03/05 04:21:35 wsun Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/plugins/L1CondDBIOVWriter.h"

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
L1CondDBIOVWriter::L1CondDBIOVWriter(const edm::ParameterSet& iConfig)
   : m_writer( iConfig.getParameter<std::string> ("offlineDB"),
	       iConfig.getParameter<std::string> ("offlineAuthentication") ),
     m_reader( iConfig.getParameter<std::string> ("offlineDB"),
	       iConfig.getParameter<std::string> ("offlineAuthentication") ),
     m_keyTag( iConfig.getParameter<std::string> ("L1TriggerKeyTag") )

{
   //now do what ever initialization is needed
   typedef std::vector<edm::ParameterSet> ToSave;
   ToSave toSave = iConfig.getParameter<ToSave> ("toPut");
   for (ToSave::const_iterator it = toSave.begin (); it != toSave.end (); it++)
   {
      std::string record = it->getParameter<std::string> ("record");
      std::string tag = it->getParameter<std::string> ("tag");

      // Copy items to the list items list
      std::map<std::string, std::string >::iterator rec =
	 m_recordToTagMap.insert( std::make_pair( record, tag ) ).first ;
   }
}


L1CondDBIOVWriter::~L1CondDBIOVWriter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1CondDBIOVWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // Get L1TriggerKeyList
   ESHandle< L1TriggerKeyList > keyList ;
   iSetup.get< L1TriggerKeyListRcd >().get( keyList ) ;

   // Get dummy L1TriggerKey -- only has TSC key, not subsystem keys
   ESHandle< L1TriggerKey > dummyKey ;
   iSetup.get< L1TriggerKeyRcd >().get( dummyKey ) ;

   unsigned long long run = iEvent.id().run() ;

   // Use TSC key and L1TriggerKeyList to find next run's L1TriggerKey token
   std::string keyToken = keyList->token( dummyKey->getTSCKey() ) ;

   // Update IOV sequence for this token with since-time = new run 
   m_writer.updateIOV( m_keyTag, keyToken, run ) ;

   // Read current L1TriggerKey directly from ORCON using token
//    L1TriggerKey key ;
//    m_reader.readPayload( keyToken, key ) ;
   L1TriggerKey key = m_reader.readKey( keyToken ) ;

   // Loop over record@type in L1TriggerKey
   L1TriggerKey::RecordToKey::const_iterator itr =
      key.recordToKeyMap().begin() ;
   L1TriggerKey::RecordToKey::const_iterator end =
      key.recordToKeyMap().end() ;

   for( ; itr != end ; ++itr )
   {
      // Find payload token
      std::string recordType = itr->first ;
      std::string subsystemKey = itr->second ;
      std::string payloadToken = keyList->token( recordType, subsystemKey ) ;
      if( payloadToken.empty() )
	{
	  throw cond::Exception( "L1CondDBIOVWriter: empty payload token" ) ;
	}
      // assert( !payloadToken.empty() ) ;

      // Extract record name from recordType
      std::string recordName( recordType, 0, recordType.find_first_of("@") ) ;

      // Find tag for IOV token
      std::map<std::string, std::string >::const_iterator recordToTagItr =
	 m_recordToTagMap.find( recordName ) ;
      if( recordToTagItr == m_recordToTagMap.end() )
	{
	  throw cond::Exception( "L1CondDBIOVWriter: no tag for record "
				 + recordName ) ;
	}
      // assert( recordToTagItr != m_recordToTagMap.end() ) ;

      m_writer.updateIOV( recordToTagItr->second, payloadToken, run ) ;
   }
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1CondDBIOVWriter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1CondDBIOVWriter::endJob() {
}

//define this as a plug-in
//DEFINE_FWK_MODULE(L1CondDBIOVWriter);
