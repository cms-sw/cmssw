#ifndef CondTolls_L1Trigger_DataReader_h
#define CondTolls_L1Trigger_DataReader_h

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
 */
class DataReader : public DataManager
{
    public:
        explicit DataReader (const std::string & connect, const std::string & catalog)
            : DataManager (connect, catalog) { }
        virtual ~DataReader () { };

        L1TriggerKey readKey (const std::string & tag, const edm::RunNumber_t & run);
        template<typename T>
        T readPayload (const L1TriggerKey & key);

        std::pair<edm::RunNumber_t, edm::RunNumber_t> interval (const std::string & tag, const edm::RunNumber_t & run);
        /* Returns pair that indictates that this interval is invalid.
         * At the moment it will return interval of (-1, -2), this is not always good
         * so one should thing about better approach
         */
        static std::pair<edm::RunNumber_t, edm::RunNumber_t> invalid ();

    protected:
        IntervalManager<edm::RunNumber_t, std::string> loadIntervals (const std::string & tag);

        /* Stores a list of intervals associated with each tag.
         * Most of the time it should be tag, but this system supports any number of them
         */
        std::map<std::string, IntervalManager<edm::RunNumber_t, std::string> > intervals;
};

/* Template functions implementation */

template<typename T>
T DataReader::readPayload (const L1TriggerKey & key)
{
    pool->connect ();
    pool->startTransaction (false);

    // so we have a key, let's use it to load data
    // Yeh, I need to load data from DB, and I do not have a payload token...
    cond::IOVService iov (*pool);
    cond::IOVIterator * iovIt = iov.newIOVIterator (metadata->getToken (key.getKey ()));
    std::string payload;

    // I do not care about validity interval as by design validity of this object is infinitive
    // only key has a validiti, also I do not care wich record to load - first, last, any.
    // Technically there should be only one
    while (iovIt->next ())
    {
        assert (payload.empty ());      // check if have not found toke already, if we did - two payloads per tag? Bad
        payload = iovIt->payloadToken ();
    }

    delete iovIt;

    assert (!payload.empty ());
    cond::Ref<T> csc (*pool, payload);

    pool->commit ();
    pool->disconnect ();
    return T (*csc);
}

}

#endif
