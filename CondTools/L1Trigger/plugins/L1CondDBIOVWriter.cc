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
// $Id: L1CondDBIOVWriter.cc,v 1.9 2008/12/15 21:41:37 wsun Exp $
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
   : m_tscKey( iConfig.getParameter<std::string> ("tscKey") ),
     m_ignoreTriggerKey( iConfig.getParameter<bool> ("ignoreTriggerKey") )
{
   //now do what ever initialization is needed
   typedef std::vector<edm::ParameterSet> ToSave;
   ToSave toSave = iConfig.getParameter<ToSave> ("toPut");
   for (ToSave::const_iterator it = toSave.begin (); it != toSave.end (); it++)
   {
      std::string record = it->getParameter<std::string> ("record");
      std::string type = it->getParameter<std::string> ("type");
      m_recordTypes.push_back( record  + "@" + type ) ;
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
       if( !m_tscKey.empty() )
	 {
	   // Use TSC key and L1TriggerKeyList to find next run's
	   // L1TriggerKey token
	   std::string keyToken = keyList->token( m_tscKey ) ;

	   // Update IOV sequence for this token with since-time = new run 
	   triggerKeyIOVUpdated =
	     m_writer.updateIOV( "L1TriggerKeyRcd", keyToken, run ) ;

	   // Read current L1TriggerKey directly from ORCON using token
	   L1TriggerKey key ;
	   m_writer.readKey( keyToken, key ) ;

	   recordTypeToKeyMap = key.recordToKeyMap() ;
	 }
       else
	 {
	   // For use with Run Settings, no corresponding L1TrigerKey in
	   // ORCON.

	   // Get L1TriggerKey from EventSetup
	   ESHandle< L1TriggerKey > esKey ;
	   iSetup.get< L1TriggerKeyRcd >().get( esKey ) ;

	   recordTypeToKeyMap = esKey->recordToKeyMap() ;
	 }
     }
   else
     {
       std::vector<std::string >::const_iterator
	 recordTypeItr = m_recordTypes.begin() ;
       std::vector<std::string >::const_iterator
	 recordTypeEnd = m_recordTypes.end() ;

       for( ; recordTypeItr != recordTypeEnd ; ++recordTypeItr )
	 {
	   recordTypeToKeyMap.insert(
	     std::make_pair( *recordTypeItr, m_tscKey ) ) ;
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
	   if( objectKey == L1TriggerKey::kNullKey )
	     {
	       edm::LogVerbatim( "L1-O2O" )
		 << "L1CondDBIOVWriter: null object key for "
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
		     "L1CondDBIOVWriter: empty payload token for " +
		     recordType + ", key " + objectKey );
		 }

	       std::string recordName( recordType,
				       0, recordType.find_first_of("@") ) ;
	       m_writer.updateIOV( recordName,
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
