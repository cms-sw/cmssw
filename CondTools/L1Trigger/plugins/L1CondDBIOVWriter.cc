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
// $Id: L1CondDBIOVWriter.cc,v 1.20 2011/05/10 19:16:56 wsun Exp $
//
//


// system include files
#include <sstream>

// user include files
#include "CondTools/L1Trigger/plugins/L1CondDBIOVWriter.h"
#include "CondTools/L1Trigger/interface/DataWriter.h"

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
     m_ignoreTriggerKey( iConfig.getParameter<bool> ("ignoreTriggerKey") ),
     m_logKeys( iConfig.getParameter<bool>( "logKeys" ) ),
     m_logTransactions( iConfig.getParameter<bool>( "logTransactions" ) ),
     m_forceUpdate( iConfig.getParameter<bool>( "forceUpdate" ) )
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
   L1TriggerKeyList keyList ;
   l1t::DataWriter dataWriter ;
   if( !dataWriter.fillLastTriggerKeyList( keyList ) )
     {
       edm::LogError( "L1-O2O" )
         << "Problem getting last L1TriggerKeyList" ;
     }

   unsigned long long run = iEvent.id().run() ;

   L1TriggerKey::RecordToKey recordTypeToKeyMap ;

   bool triggerKeyIOVUpdated = true ;

   // Start log string, convert run number into string
   std::stringstream ss ;
   ss << run ;
   std::string log = "KEYLOG runNumber=" + ss.str() ;
   bool logRecords = true ;

   if( !m_ignoreTriggerKey )
     {
       if( !m_tscKey.empty() )
	 {
           edm::LogVerbatim( "L1-O2O" )
             << "Object key for L1TriggerKey@L1TriggerKeyRcd: "
             << m_tscKey ;

	   // Use TSC key and L1TriggerKeyList to find next run's
	   // L1TriggerKey token
	   std::string keyToken = keyList.token( m_tscKey ) ;

	   // Update IOV sequence for this token with since-time = new run 
	   triggerKeyIOVUpdated =
	     m_writer.updateIOV( "L1TriggerKeyRcd", keyToken, run, m_logTransactions ) ;

	   // Read current L1TriggerKey directly from ORCON using token
	   L1TriggerKey key ;
	   m_writer.readObject( keyToken, key ) ;

	   recordTypeToKeyMap = key.recordToKeyMap() ;

           // Replace spaces in key with ?s.  Do reverse substitution when
           // making L1TriggerKey.
	   std::string tmpKey = m_tscKey ;
           replace( tmpKey.begin(), tmpKey.end(), ' ', '?' ) ;
           log += " tscKey=" + tmpKey ;
	   logRecords = false ;
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
   bool throwException = false ;

   if( triggerKeyIOVUpdated || m_forceUpdate )
     {
       // Loop over record@type in L1TriggerKey
       L1TriggerKey::RecordToKey::const_iterator itr =
	 recordTypeToKeyMap.begin() ;
       L1TriggerKey::RecordToKey::const_iterator end =
	 recordTypeToKeyMap.end() ;

       for( ; itr != end ; ++itr )
	 {
	   std::string recordType = itr->first ;
	   std::string objectKey = itr->second ;

	   std::string recordName( recordType,
				   0, recordType.find_first_of("@") ) ;

	   if( logRecords )
	     {
	       // Replace spaces in key with ?s.  Do reverse substitution when
	       // making L1TriggerKey.
	       std::string tmpKey = objectKey ;
	       replace( tmpKey.begin(), tmpKey.end(), ' ', '?' ) ;
	       log += " " + recordName + "Key=" + tmpKey ;
	     }

	   // Do nothing if object key is null.
	   if( objectKey == L1TriggerKey::kNullKey )
	     {
	       edm::LogVerbatim( "L1-O2O" )
		 << "L1CondDBIOVWriter: null object key for "
		 << recordType << "; skipping this record." ;
	     }
	   else
	     {
	       // Find payload token
               edm::LogVerbatim( "L1-O2O" )
                 << "Object key for "
                 << recordType << ": " << objectKey ;

	       std::string payloadToken = keyList.token( recordType,
							 objectKey ) ;
	       if( payloadToken.empty() )
		 {
		   edm::LogVerbatim( "L1-O2O" )
		     << "L1CondDBIOVWriter: empty payload token for " +
		     recordType + ", key " + objectKey ;

		   throwException = true ;
		 }
	       else
		 {
		   m_writer.updateIOV( recordName,
				       payloadToken,
				       run,
				       m_logTransactions ) ;
		 }
	     }
	 }
     }

   if( m_logKeys )
     {
       edm::LogVerbatim( "L1-O2O" ) << log ;
     }

   if( throwException )
     {
       throw cond::Exception( "L1CondDBIOVWriter: empty payload tokens" ) ;
     }
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1CondDBIOVWriter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1CondDBIOVWriter::endJob() {
}

//define this as a plug-in
//DEFINE_FWK_MODULE(L1CondDBIOVWriter);
