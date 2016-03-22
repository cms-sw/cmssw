#ifndef CondFormats_L1TObjects_L1TriggerKeyListExt_h
#define CondFormats_L1TObjects_L1TriggerKeyListExt_h

// system include files
#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <map>

class L1TriggerKeyListExt
{

   public:
      L1TriggerKeyListExt();
      virtual ~L1TriggerKeyListExt();

      typedef std::map< std::string, std::string > KeyToToken ;
      typedef std::map< std::string, KeyToToken > RecordToKeyToToken ;

      // ---------- const member functions ---------------------

      // Get payload token for L1TriggerKeyExt
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

      // Get TSC key for a given L1TriggerKeyExt payload token
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
      //L1TriggerKeyListExt(const L1TriggerKeyListExt&); // stop default

      //const L1TriggerKeyListExt& operator=(const L1TriggerKeyListExt&); // stop default

      // ---------- member data --------------------------------

      // map of TSC key (first) to L1TriggerKeyExt payload token (second)
      KeyToToken m_tscKeyToToken ;

      // map of subsystem key (second/first) to configuration data payload
      // token (second/second), keyed by record@type (first)
      RecordToKeyToToken m_recordKeyToken ;

  COND_SERIALIZABLE;
};


#endif
