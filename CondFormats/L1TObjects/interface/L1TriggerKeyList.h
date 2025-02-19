#ifndef CondFormats_L1TObjects_L1TriggerKeyList_h
#define CondFormats_L1TObjects_L1TriggerKeyList_h
// -*- C++ -*-
//
// Package:     L1TObjects
// Class  :     L1TriggerKeyList
// 
/**\class L1TriggerKeyList L1TriggerKeyList.h CondFormats/L1TObjects/interface/L1TriggerKeyList.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Fri Feb 29 20:44:53 CET 2008
// $Id: L1TriggerKeyList.h,v 1.4 2009/04/06 01:58:49 wsun Exp $
//

// system include files
#include <string>
#include <map>

// user include files

// forward declarations

class L1TriggerKeyList
{

   public:
      L1TriggerKeyList();
      virtual ~L1TriggerKeyList();

      typedef std::map< std::string, std::string > KeyToToken ;
      typedef std::map< std::string, KeyToToken > RecordToKeyToToken ;

      // ---------- const member functions ---------------------

      // Get payload token for L1TriggerKey
      std::string token( const std::string& tscKey ) const ;

      // Get payload token for configuration data
      std::string token( const std::string& recordName,
			 const std::string& dataType,
			 const std::string& key ) const ;

      // Get payload token for configuration data
      std::string token( const std::string& recordType, // "record@type"
			 const std::string& key ) const ;

      const KeyToToken& tscKeyToTokenMap() const
	{ return m_tscKeyToToken ; }

      const RecordToKeyToToken& recordTypeToKeyToTokenMap() const
	{ return m_recordKeyToken ; }

      // Get object key for a given payload token.  In practice, each
      // record in the CondDB has only one object, so there is no need to
      // specify the data type.
      std::string objectKey( const std::string& recordName,
			     const std::string& payloadToken ) const ;

      // Get TSC key for a given L1TriggerKey payload token
      std::string tscKey( const std::string& triggerKeyPayloadToken ) const ;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      // Store payload token for L1TriggerKey, return true if successful
      bool addKey( const std::string& tscKey,
		   const std::string& payloadToken,
		   bool overwriteKey = false ) ;

      // Store payload token for configuration data, return true if successful
      bool addKey( const std::string& recordType, // "record@type"
		   const std::string& key,
		   const std::string& payloadToken,
		   bool overwriteKey = false ) ;

   private:
      //L1TriggerKeyList(const L1TriggerKeyList&); // stop default

      //const L1TriggerKeyList& operator=(const L1TriggerKeyList&); // stop default

      // ---------- member data --------------------------------

      // map of TSC key (first) to L1TriggerKey payload token (second)
      KeyToToken m_tscKeyToToken ;

      // map of subsystem key (second/first) to configuration data payload
      // token (second/second), keyed by record@type (first)
      RecordToKeyToToken m_recordKeyToken ;
};


#endif
