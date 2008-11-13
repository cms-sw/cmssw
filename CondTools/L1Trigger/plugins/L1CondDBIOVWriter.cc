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
// $Id: L1CondDBIOVWriter.cc,v 1.5 2008/09/27 02:38:19 wsun Exp $
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
     m_tscKey( iConfig.getParameter<std::string> ("tscKey") ),
     m_keyTag( iConfig.getParameter<std::string> ("L1TriggerKeyTag") ),
     m_ignoreTriggerKey( iConfig.getParameter<bool> ("ignoreTriggerKey") )
{
   //now do what ever initialization is needed
   typedef std::vector<edm::ParameterSet> ToSave;
   ToSave toSave = iConfig.getParameter<ToSave> ("toPut");
   for (ToSave::const_iterator it = toSave.begin (); it != toSave.end (); it++)
   {
      std::string record = it->getParameter<std::string> ("record");
      std::string type = it->getParameter<std::string> ("type");
      std::string tag = it->getParameter<std::string> ("tag");

      // Copy items to the list items list
      m_recordTypeToTagMap.insert( std::make_pair( record + "@" + type,
						   tag ) ) ;
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

   unsigned long long run = iEvent.id().run() ;

   L1TriggerKey::RecordToKey recordTypeToKeyMap ;

   bool triggerKeyIOVUpdated = true ;
   if( !m_ignoreTriggerKey )
     {
       // Do nothing if TSC key is null
       if( !m_tscKey.empty() )
	 {
	   // Use TSC key and L1TriggerKeyList to find next run's L1TriggerKey
	   // token
	   std::string keyToken = keyList->token( m_tscKey ) ;

	   // Update IOV sequence for this token with since-time = new run 
	   triggerKeyIOVUpdated =
	     m_writer.updateIOV( m_keyTag, keyToken, run ) ;

	   // Read current L1TriggerKey directly from ORCON using token
	   L1TriggerKey key = m_reader.readKey( keyToken ) ;

	   recordTypeToKeyMap = key.recordToKeyMap() ;
	 }
     }
   else
     {
       std::map<std::string, std::string >::const_iterator recordTypeToTagItr =
	 m_recordTypeToTagMap.begin() ;
       std::map<std::string, std::string >::const_iterator recordTypeToTagEnd =
	 m_recordTypeToTagMap.end() ;

       for( ; recordTypeToTagItr != recordTypeToTagEnd ; ++recordTypeToTagItr )
	 {
	   recordTypeToKeyMap.insert(
	     std::make_pair( recordTypeToTagItr->first, m_tscKey ) ) ;
	 }
     }

   // If L1TriggerKey IOV was already up to date, then so are all its
   // sub-records.
   if( triggerKeyIOVUpdated )
     {
       // Loop over record@type in L1TriggerKey
       L1TriggerKey::RecordToKey::const_iterator itr =
	 recordTypeToKeyMap.begin() ;
       L1TriggerKey::RecordToKey::const_iterator end =
	 recordTypeToKeyMap.end() ;

       for( ; itr != end ; ++itr )
	 {
	   // Do nothing if object key is null.
	   std::string recordType = itr->first ;
	   std::string objectKey = itr->second ;
	   if( objectKey.empty() )
	     {
	       edm::LogVerbatim( "L1-O2O" )
		 << "L1CondDBIOVWriter: empty object key for "
		 << recordType << "; skipping this record." ;
	     }
	   else
	     {
	       // Find payload token
	       std::string payloadToken = keyList->token( recordType,
							  objectKey ) ;
	       if( payloadToken.empty() )
		 {
		   throw cond::Exception(
		     "L1CondDBIOVWriter: empty payload token" );
		 }
	       // assert( !payloadToken.empty() ) ;

	       // Find tag for IOV token
	       std::map<std::string, std::string >::const_iterator
		 recordTypeToTagItr =
		 m_recordTypeToTagMap.find( recordType ) ;
	       if( recordTypeToTagItr == m_recordTypeToTagMap.end() )
		 {
		   throw cond::Exception(
		     "L1CondDBIOVWriter: no tag for record@type " +
		     recordType ) ;
		 }

	       m_writer.updateIOV( recordTypeToTagItr->second,
				   payloadToken,
				   run ) ;
	     }
	 }
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
