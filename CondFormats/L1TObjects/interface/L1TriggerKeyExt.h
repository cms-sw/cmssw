#ifndef CondFormats_L1TObjects_L1TriggerKeyExt_h
#define CondFormats_L1TObjects_L1TriggerKeyExt_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <map>

/* L1 key used to load all other configuration data from offline db.
 * This class is just a proxy to the real data. It will contain mapping from data and record
 * pair to the payload token that could be used to read data. So the use case could be as follows:
 *   1. User read L1TriggerKey for given Tag and IOV pair.
 *   2. For each record and type that user whant to load, it ask method get for the payload.
 *   3. Reads the data with payloads extracted from step 2.
 *
 * It is not adviced for user to use this class and direct Pool DB manipulation. One should use
 * DataReader and DataWriter classes.
 *
 * The good point to note is that IOV of all L1 trigger condfiguration is controled bay IOV of L1TriggeKey.
 * If new configuration has to be created - new L1TriggerKey has to be saved/loaded. More then one key can use
 * the same paylaod token. This would just mean that data pointed by this payload token has not changed.
 */
class L1TriggerKeyExt
{
public:
    typedef std::map<std::string, std::string> RecordToKey;

    enum L1Subsystems
      {
	kuGT,
	kuGMT,
	kCALO,
	kEMTF,
	kOMTF,
	kBMTF,
	kTWINMUX,
	kNumberSubsystems
      } ;

    // Empty strings cannot be stored in the CondDB, so define a null key string.
    const static std::string kNullKey ;

    const static std::string kEmptyKey ;

    // Constructors
    L1TriggerKeyExt ()
      {
	for( int i = 0 ; i < kNumberSubsystems ; ++i )
	  {
	    m_subsystemKeys[ i ] = kNullKey ;
	  }
      }

    /* Adds new record and type mapping to payload. If such exists, nothing happens */
    void add (const std::string & record, const std::string & type, const std::string & key)
    { m_recordToKey.insert (std::make_pair (record + "@" + type, key.empty() ? kNullKey : key)); }

    void add (const RecordToKey& map)
    {
      for( RecordToKey::const_iterator itr = map.begin() ;
	   itr != map.end() ;
	   ++itr )
	{
	  m_recordToKey.insert( std::make_pair( itr->first, itr->second.empty() ? kNullKey : itr->second ) ) ;
	}
    }

    void setTSCKey( const std::string& tscKey )
    { m_tscKey = tscKey ; }

    void setSubsystemKey( L1Subsystems subsystem, const std::string& key )
    { m_subsystemKeys[ subsystem ] = key.empty() ? kNullKey : key ; }

    /* Gets payload key for record and type. If no such paylaod exists, emtpy string
     * is returned.
     */
    std::string get (const std::string & record, const std::string & type) const
    {
        RecordToKey::const_iterator it = m_recordToKey.find (record + "@" + type);
        if (it == m_recordToKey.end ())
            return std::string ();
        else
	  return it->second == kNullKey ? kEmptyKey : it->second ;
    }

    const std::string& tscKey() const
      { return m_tscKey ; }

    const std::string& subsystemKey( L1Subsystems subsystem ) const
      { std::map<int,std::string>::const_iterator key = m_subsystemKeys.find( subsystem );
        return key == m_subsystemKeys.end() || key->second == kNullKey ? kEmptyKey : key->second ; }

    // NB: null keys are represented by kNullKey, not by an empty string
    const RecordToKey& recordToKeyMap() const
      { return m_recordToKey ; }

protected:
    /* Mapping from records and types to tokens.
     * I as unvable to make type std::map<std::pair<std::string, std::string>, std::string> persistent
     * so record and type are concatanated with @ sign and resulting string is used as a key.
     */

  // wsun 03/2008: instead of tokens, store the configuration keys instead.
/*     typedef std::map<std::string, std::string> RecordsToToken; */
/*     RecordsToToken recordsToToken; */
    RecordToKey m_recordToKey;


    // wsun 03/2008: add data member for TSC key
    std::string m_tscKey ;
    std::map<int,std::string> m_subsystemKeys;

  COND_SERIALIZABLE;
};

#endif
