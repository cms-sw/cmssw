#include "CondTools/L1Trigger/src/DataWriter.h"

#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"

#include <utility>

namespace l1t
{

DataWriter::DataWriter (const std::string & connect, const std::string & catalog)
{
    // Initialize session used with this object
    session = new cond::DBSession ();
    session->sessionConfiguration ().setAuthenticationMethod (cond::Env);
//    session->sessionConfiguration ().setMessageLevel (cond::Info);
    session->connectionConfiguration ().enableConnectionSharing ();
    session->connectionConfiguration ().enableReadOnlySessionOnUpdateConnections ();

    // create Coral connection and pool. This ones should be connected on required basis
    coral = new cond::RelationalStorageManager (connect, session);
    pool = new cond::PoolStorageManager (connect, catalog, session);

    // and data object
    metadata = new cond::MetaData (*coral);
}


DataWriter::~DataWriter ()
{
    // detele all in reverse direction
    // closing and comminting may be a good idea here
    // in case we have exception in some other part between connect/close pairs

    delete metadata;
    delete pool;
    delete coral;
    delete session;
}

/* Data writting functions */

std::string DataWriter::findTokenForTag (const std::string & tag)
{
    // fast check in cashe
    TagToToken::const_iterator found = tagToToken.find (tag);
    if (found != tagToToken.end ())
        return found->second;

    // else slow way, also we may need to create db
    coral->connect (cond::ReadWriteCreate);
    coral->startTransaction (false);

    std::string tagToken;
    if (metadata->hasTag (tag))
        tagToken = metadata->getToken (tag);

    coral->commit ();
    coral->disconnect ();

    // if not found empty string is returned
    return tagToken;
}

void DataWriter::writeKey (L1TriggerKey * key, const std::string & tag, const int sinceRun)
{
    // writting key as bit more complicated. At this time we have to worry
    // about such things if the key already exists or not
    // Also we need to get IOVToken for given tag if key exist
    // if it does not, then we need to addMapping in the end

    // Bad part - get TagTokent for given tag.
    // We use layzy cash to save all tag adn tokens, in case we save key with same tag
    std::string tagToken = findTokenForTag (tag);
    bool requireMapping = tagToken.empty ();

    // work with pool db
    pool->connect ();
    pool->startTransaction (false);

    cond::Ref<L1TriggerKey> ref (*pool, key);
    ref.markWrite ("L1TriggerKeyRcd");

    cond::IOVService iov (*pool);

    // Create editor, with or wothoug TagToken
    cond::IOVEditor * editor;
    editor = iov.newIOVEditor (tagToken);

    // finally insert new IOV
    editor->insert (sinceRun, ref.token ());
    tagToken = editor->token ();
    delete editor;

    pool->commit ();
    pool->disconnect ();

    if (tagToToken.find (tag) != tagToToken.end ())
        tagToToken.insert (std::make_pair (tag, tagToken));

    // Assign payload token with IOV value
    if (requireMapping)
        addMappings (tag, tagToken);
}

void DataWriter::addMappings (const std::string tag, const std::string iovToken)
{
    coral->connect (cond::ReadWriteCreate);
    coral->startTransaction (false);

    metadata->addMapping (tag, iovToken);

    coral->commit ();
    coral->disconnect ();
}

} // ns
