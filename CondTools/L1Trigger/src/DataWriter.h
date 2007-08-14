#ifndef CONDTOOLS_L1TRIGER_DAtAWRITER_H
#define CONDTOOLS_L1TRIGER_DAtAWRITER_H

// Framework
#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/Ref.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"

// L1T includes
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondTools/L1Trigger/src/DataManager.h"

#include <string>
#include <map>

namespace l1t
{

/* This is a helper class that will allow user to write data to database.
 * Class provides managament of all connection and catalogs. As well as managing
 * key and payload relationships.
 */
class DataWriter : public DataManager
{
    public:
        explicit DataWriter (const std::string & connect, const std::string & catalog)
            : DataManager (connect, catalog) {};
        virtual ~DataWriter () {};

        /* Data writting functions */

        /* Writes payload to DB and assignes it to provided L1TKey.
         */
        template<typename T>
        void writePayload (const L1TriggerKey & key, T * payload, const std::string & recordName);
        /* Writes given key to DB and starts its IOV from provided run number.
         * From here pointer ownership is managed by POOL
         */
        void writeKey (L1TriggerKey * key, const std::string & tag, const int sinceRun);

    protected:
        void addMappings (const std::string tag, const std::string iovToken);
        std::string findTokenForTag (const std::string & tag);

        // map to speed up detection if we already have mapping from given tag to token
        typedef std::map<std::string, std::string> TagToToken;
        TagToToken tagToToken;
};

/* Part of the implementation: Template functions that can't go to cc failes */

template<typename T>
void DataWriter::writePayload (const L1TriggerKey & key, T * payload, const std::string & recordName)
{
    pool->connect ();
    pool->startTransaction (false);

    cond::Ref<T> ref (*pool, payload);
    ref.markWrite (recordName);

    // we know that this is new IOV token so we can skip test. Execption at this moment
    // is also ok, becaus it is not knwo what to do if we try to reuse token
    // However, exceptions will be rised from addMappings method
    cond::IOVService iov (*pool);
    cond::IOVEditor * editor = iov.newIOVEditor ();
    editor->insert (edm::IOVSyncValue::endOfTime ().eventID ().run (), ref.token ());
    std::string token = editor->token ();
    delete editor;

    pool->commit ();
    pool->disconnect ();

    // Assign payload token with IOV value
    addMappings (key.getKey (), token);
}

} // ns

#endif
