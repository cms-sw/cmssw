#ifndef CondTolls_L1Trigger_DataReader_h
#define CondTolls_L1Trigger_DataReader_h

#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/DataKey.h"

#include "DataFormats/Provenance/interface/RunID.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"

#include "CondTools/L1Trigger/interface/DataManager.h"
#include "CondTools/L1Trigger/interface/Interval.h"

#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/PluginSystem/interface/DataProxy.h"

#include "boost/shared_ptr.hpp"
#include <string>
#include <vector>
#include <map>

namespace l1t
{

/* This class is used to read L1 trigger configuration data from pool DB. It will provide
 * service for managing connections and loading all the data. This class should be
 * used with l1t::DataWriter
 *
 * This class provides two different access method. One is used when user knows exact types to load
 * from DB at compile time. In such case she or he should use methods readKey (...) and readPayload (...).
 * 
 * In case user is running in DataProxyProvider mode and does not know all the types at compile time,
 * she or he should use methods that returns DataProxy. It is assumed that these proxies should be then passed
 * to frameork, although this requirement is not enforced.
 *
 * For all payload read operations user has to provide a valid L1TriggerKey. It is expected that this key comes
 * from readKey method, although, if user is able to guarante that key is valid it can use any other source.
 *
 * Validity of all data is controled by validity of L1TriggerKey.
 */
class DataReader : public DataManager
{
    public:
        /* Constructors. This system will use pool db that is configured via provided parameters.
         * connect - connection string to pool db, e.g. sqlite_file:test.db
         * catalog - catalog that should be used for this connection. e.g. file:test.xml
         */
  //explicit DataReader (const std::string & connect, const std::string & catalog);
  explicit DataReader (const std::string& connectString,
		       const std::string& authenticationPath );
        virtual ~DataReader ();

        /* Returns key for givent tag ant IOV time moment. This key then can be used
         * to load all other data
         */
        L1TriggerKey readKey (const std::string & tag, const edm::RunNumber_t & run);
        /* Reads payload from the DB. This one requires L1 key to figure out which record is considered
         * valid
         */

        L1TriggerKey readKey (const std::string & payloadToken);

        template<typename T>
        T readPayload (const L1TriggerKey & key, const std::string & record);

      // Read payload by payload token
      template<typename T>
      void readPayload ( const std::string & payloadToken, T& payload );

      //void readPayload ( const std::string & payloadToken, L1TriggerKey& payload );

        /* Returns proxy that should provided objects of given type for given object. This will not
         * take any information about IOV moment. You can set this information via updateToken.
         * Also, one should note that when object is returned by this method it is not valid
         * until updateToken is called for this record and type.
         * 
         * Such strange behavior requirements comes from the fact how DataProxyProvider works. This method
         * is designed, but not limited, to be used by DataProxyProvider (e.g. L1TDBESSource)
         */
        std::pair<boost::shared_ptr<edm::eventsetup::DataProxy>, edm::eventsetup::DataKey>
        payload (const std::string & record, const std::string & type);

        /* Returns proxy that is responsible for creating L1TriggerKey record and type itself.
         * All requirements and notes are the same as in payload (...) method.
         */
        std::pair<boost::shared_ptr<edm::eventsetup::DataProxy>, edm::eventsetup::DataKey> key ();

        /* Returns IOV interval for given key tag and IOV time moment. May return invalid() interval if time is
         * not found.
         */
        std::pair<edm::RunNumber_t, edm::RunNumber_t> interval (const std::string & tag, const edm::RunNumber_t & run);

        /* Returns pair that indictates that this interval is invalid.
         * At the moment it will return interval of (-1, -2), this is not always good
         * so one should thing about better approach
         */
        static std::pair<edm::RunNumber_t, edm::RunNumber_t> invalid ();

        /* Instructs previusly returned proxy (from method payload (...)) to load data
         * associated with given tag and IOV value. Thus, record and type parameters are used to identify
         * which proxy to update. Correct record and type is selected from DB based on proovided key
         */
        void updateToken (const std::string & record, const std::string & type,
            const L1TriggerKey & key);

        /* Updates key proxy (returned by method key (...)) to start producing new key with given tag and
         * run number. This will not effect objects loaded with method payload (...). So one has to call other version
         * of updateToken to make sure than payload is returned correctly
         */
        void updateToken (const std::string & tag, const edm::RunNumber_t run);

    protected:
        /* Loads all IOV intervals that are valid in DB for given tag.
         */
        IntervalManager<edm::RunNumber_t, std::string> loadIntervals (const std::string & tag);

        /* Returns payload token for given tag and run number
         */
        std::string payloadToken (const std::string & tag, const edm::RunNumber_t run) const;

        /* Helper method that does the actual work for method payload (...) and key (...)
         */
        std::pair<boost::shared_ptr<edm::eventsetup::DataProxy>, edm::eventsetup::DataKey>
        createPayload (const std::string & record, const std::string & type);

        /* Stores a list of intervals associated with each tag.
         * Most of the time it should be tag, but this system supports any number of them
         */
        std::map<std::string, IntervalManager<edm::RunNumber_t, std::string> > intervals;

        /* I am using DataProxy. Its construtor takes a last parameteres as an iterator
         * to map of string of string. Technically it needs only the second string.
         * The problem that I have here is that I have to make sure that DataProxy that I am
         * returning will die before map whos iterator it uses. Thus we have map here.
         */
        std::map<std::string, std::string> typeToToken;
};

/* Template functions implementation */

template<typename T>
T DataReader::readPayload (const L1TriggerKey & key, const std::string & record)
{
/*     pool->connect (); */
  //connection->connect( session ) ;
/*     pool->startTransaction (false); */
  cond::PoolTransaction& pool = connection->poolTransaction() ;
  pool.start( false ) ;

    // Convert type to class name.
    std::string typeName = 
        edm::eventsetup::heterocontainer::HCTypeTagTemplate<T, edm::eventsetup::DataKey>::className ();
    std::string token = key.get (record, typeName);
    assert (!token.empty ());

    cond::TypedRef<T> ref (pool, token);

    pool.commit ();
/*     pool->disconnect (); */
    //connection->disconnect ();
    return T (*ref);
}

template<typename T>
void DataReader::readPayload( const std::string& payloadToken, T& payload )
{
   cond::PoolTransaction& pool = connection->poolTransaction() ;
   pool.start( true ) ;

   cond::TypedRef<T> ref (pool, payloadToken);

   pool.commit ();
   payload = *ref ;
}

}

#endif
