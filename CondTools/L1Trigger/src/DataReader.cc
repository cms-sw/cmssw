#include "CondTools/L1Trigger/src/DataReader.h"

#include "CondCore/PluginSystem/interface/DataProxy.h"
#include "CondCore/PluginSystem/interface/ProxyFactory.h"

#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVIterator.h"

#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

namespace l1t
{
    
DataReader::DataReader (const std::string & connect, const std::string & catalog)
    : DataManager (connect, catalog)
{
    // we always maintain pool connection open, so that DataProxy could load the data.
    pool->connect ();
}

DataReader::~DataReader ()
{
    pool->disconnect ();
}

L1TriggerKey DataReader::readKey (const std::string & tag, const edm::RunNumber_t & run)
{
    pool->startTransaction (true);

    // get interval for given time, and laod intervals fromd DB if they do not exist for the tag
    if (intervals.find (tag) == intervals.end ())
        intervals.insert (std::make_pair (tag, loadIntervals (tag)));

    // get correct interval and laod the key for it, we could use interval (...) method, but it returns
    // a bit different type, then I need here. I.e. no payload, only start and end times.
    Interval<edm::RunNumber_t, std::string> data = intervals [tag].find (run);
    if (data == Interval<edm::RunNumber_t, std::string>::invalid ())
        throw cond::Exception ("DataReader::readKey") << "Provided time value " << run << " does not exist in the database "
            << "for l1 tag \"" << tag << "\"";

    cond::Ref<L1TriggerKey> key (*pool, data.payload ());
    L1TriggerKey ret (*key); // clone key, so that it could be returned nicely

    pool->commit ();

    return ret;
}

IntervalManager<edm::RunNumber_t, std::string> DataReader::loadIntervals (const std::string & tag)
{
    // convert tag to tag token
    coral->connect (cond::ReadOnly);
    coral->startTransaction (true);
    std::string tagToken = metadata->getToken (tag);
    coral->commit();
    coral->disconnect ();

    pool->startTransaction (false);

    intervals.clear ();

    cond::IOVService iov (*pool);
    cond::IOVIterator * iovIt = iov.newIOVIterator (tagToken);
    IntervalManager<edm::RunNumber_t, std::string> intervals;

    // list all intervals and add to intervals manager
    while (iovIt->next ())
        intervals.addInterval (Interval<edm::RunNumber_t, std::string> (iovIt->validity ().first, iovIt->validity ().second,
                    iovIt->payloadToken ()));

    delete iovIt;
    pool->commit ();

    return intervals;
}

std::pair<edm::RunNumber_t, edm::RunNumber_t> DataReader::invalid ()
{
    static std::pair<edm::RunNumber_t, edm::RunNumber_t> invalidPair = std::pair<edm::RunNumber_t, edm::RunNumber_t> (10, 9);
    return invalidPair;
}

std::pair<edm::RunNumber_t, edm::RunNumber_t> DataReader::interval (const std::string & tag, const edm::RunNumber_t & run)
{
    // if we do not have tag in our small cache, insert it into it
    if (intervals.find (tag) == intervals.end ())
        intervals.insert (std::make_pair (tag, loadIntervals (tag)));

    Interval<edm::RunNumber_t, std::string> interval = intervals [tag].find (run);
    if (interval != Interval<edm::RunNumber_t, std::string>::invalid ())
        return std::make_pair (interval.start (), interval.end ());
    else
        // not nice, but I have no Idea how to make nicer
        return invalid ();
}

std::string DataReader::payloadToken (const std::string & tag, const edm::RunNumber_t run) const
{
    coral->connect (cond::ReadOnly);
    coral->startTransaction (true);
    std::string payload;

    std::string token = metadata->getToken (tag);

    coral->commit ();
    coral->disconnect ();

    pool->startTransaction (true);
    cond::IOVService iov (*pool);
    cond::IOVIterator * iovIt = iov.newIOVIterator (token);

    // If this is a key, then validity interval is important. But if this is a record
    // then I do not care about validity intervals. However, in case in the future they should be
    // extendet, this code should still work.
    while (iovIt->next () && payload.empty ())
        if (iovIt->validity ().first <= run && iovIt->validity ().second >= run)
            payload = iovIt->payloadToken ();

    delete iovIt;
    pool->commit ();

    assert (!payload.empty ());
    return payload;
}

/* The following code is taken from PoolDBESSource */
static std::string buildName( const std::string& iRecordName, const std::string& iTypeName )
{
  return iRecordName+"@"+iTypeName+"@Proxy";
}

void DataReader::updateToken (const std::string & record, const std::string & type,
    const L1TriggerKey & key)
{
     std::stringstream ss;
     ss << record << "@" << type;

     // find entry in the map and change its second argument - payload token
     // DataProxy will make sure to take into account this change
     std::map <std::string, std::string>::iterator it = typeToToken.find (ss.str ());
     if (it != typeToToken.end ())
         it->second = key.get (record, type);
}

void DataReader::updateToken (const std::string & tag, const edm::RunNumber_t run)
{
     std::map <std::string, std::string>::iterator it = typeToToken.find ("L1TriggerKeyRcd@L1TriggerKey");
     if (it != typeToToken.end ())
         it->second = payloadToken (tag, run);
}

std::pair<boost::shared_ptr<edm::eventsetup::DataProxy>, edm::eventsetup::DataKey>
DataReader::createPayload (const std::string & record, const std::string & type)
{
     edm::eventsetup::TypeTag typeTag = findType (type);
     std::stringstream ss;
     ss << record << "@" << type;

     std::map <std::string, std::string>::iterator it = typeToToken.find (ss.str ());

     // we do not insert any valid token. Token will be updated when updateToken is called.
     // In other words, returned record is not valid until one calls updateToken for this type
     // record
     if (it == typeToToken.end ())
         it = typeToToken.insert (std::make_pair (ss.str (), "")).first;

     boost::shared_ptr<edm::eventsetup::DataProxy> proxy(
             cond::ProxyFactory::get()->create(buildName(record, type), pool, it));

     if(0 == proxy.get())
        throw cond::Exception ("DataReader::createPayload") << "Proxy for type " << type
            << " and record " << record << " returned null";

     edm::eventsetup::DataKey dataKey (typeTag, "");
     return std::make_pair (proxy, dataKey);
}

std::pair<boost::shared_ptr<edm::eventsetup::DataProxy>, edm::eventsetup::DataKey>
DataReader::payload (const std::string & record, const std::string & type)
{
    return createPayload (record, type);
}

std::pair<boost::shared_ptr<edm::eventsetup::DataProxy>, edm::eventsetup::DataKey>
DataReader::key ()
{
    return createPayload ("L1TriggerKeyRcd", "L1TriggerKey");
}

}
