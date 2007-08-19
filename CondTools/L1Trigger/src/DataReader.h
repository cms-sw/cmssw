#ifndef CondTolls_L1Trigger_DataReader_h
#define CondTolls_L1Trigger_DataReader_h

#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/DataKey.h"

#include "CondCore/PluginSystem/interface/DataProxy.h"

#include "DataFormats/Provenance/interface/RunID.h"

#include "CondTools/L1Trigger/src/DataManager.h"
#include "CondTools/L1Trigger/src/Interval.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"

#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/DBCommon/interface/Ref.h"


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
 * to framork, although this requirement is not enforced.
 */
class DataReader : public DataManager
{
    public:
        explicit DataReader (const std::string & connect, const std::string & catalog);
        virtual ~DataReader ();

        /* Returns key for givent tag ant IOV time moment. This key then can be used
         * to load all other data
         */
        L1TriggerKey readKey (const std::string & tag, const edm::RunNumber_t & run);
        /* Reads payload from the DB. This one requires L1 key to figure out which record is considered
         * valid
         */
        template<typename T>
        T readPayload (const L1TriggerKey & key);

        /* Returns proxy that should provided oobjects of given type for given object. This will not
         * take any information about IOV moment. You can set this information via updateToken.
         * Also, one should note that when object is returned by this method it is not valid
         * until updateTOken is called for this record and type.
         * Such strange behavior requirements comes from the fact how DataProxyProvider works
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

        /* Instructs previusly returned proxy (from methods payload (...) and key (...)) to load data
         * associated with given tag and IOV value. Thus, record and type parameters are used to identify
         * which proxy to update and tag and run number what payloadToken to provide for them.
         */
        void updateToken (const std::string & record, const std::string & type,
            const std::string & tag, const edm::RunNumber_t run);

    protected:
        /* Loads all IOV intervals that are valid in DB for given tag.
         */
        IntervalManager<edm::RunNumber_t, std::string> loadIntervals (const std::string & tag);

        /* Returns payload token for given tag and run number
         */
        std::string payloadToken (const std::string & tag, const edm::RunNumber_t run = 1) const;

        /* Helper method that does teh actual work for method payload (...) and key (...)
         */
        std::pair<boost::shared_ptr<edm::eventsetup::DataProxy>, edm::eventsetup::DataKey>
        createPayload (const std::string & record, const std::string & type);

        /* Returns type object for provided type name
         */
        edm::eventsetup::TypeTag findType (const std::string & type) const;

        /* Stores a list of intervals associated with each tag.
         * Most of the time it should be tag, but this system supports any number of them
         */
        std::map<std::string, IntervalManager<edm::RunNumber_t, std::string> > intervals;

        /* I am using DataProxy. Its construtor takes a last parameteres as an iterator
         * to map of string of string. Technically it needs only the second string.
         * The problem that I have here is that I have to make sure that DataProxy that I am
         * reuturning will die before map whos iterator it uses.
         * Thus we have map here
         */
        std::map<std::string, std::string> typeToToken;
};

/* Template functions implementation */

template<typename T>
T DataReader::readPayload (const L1TriggerKey & key)
{
    pool->connect ();
    pool->startTransaction (false);

    // so we have a key, let's use it to load data
    // Yeh, I need to load data from DB, and I do not have a payload token...
    cond::Ref<T> csc (*pool, payloadToken (key.getKey ()));

    pool->commit ();
    pool->disconnect ();
    return T (*csc);
}

}

#endif
