#include "CondTools/L1Trigger/src/DataReader.h"

namespace l1t
{

L1TriggerKey DataReader::readKey (const std::string & tag, const edm::RunNumber_t & run)
{
    pool->connect ();
    pool->startTransaction (false);

    // we can use any date from the interval provided by iRecord
    Interval<edm::RunNumber_t, std::string> data = intervals [tag].find (run);
    if (data == Interval<edm::RunNumber_t, std::string>::invalid ())
        throw cond::Exception ("DataReader::readKey") << "Provided time value " << run << " does not exist in the database "
            << "for l1 tag " << tag;

    cond::Ref<L1TriggerKey> key (*pool, data.payload ());
    L1TriggerKey ret (key->getKey ());

    pool->commit ();
    pool->disconnect ();

    // first of all I need to load key from the "current" IOV
    // based on this key, I have to extract and return all other objects that are associated
    // this key and have valid (most likely infinitive IOV)
    return ret;
}

IntervalManager<edm::RunNumber_t, std::string> DataReader::loadIntervals (const std::string & tag)
{
    // convert tag to tag token
    coral->connect (cond::ReadOnly);
    coral->startTransaction (true);
    // we read tag token here. If we do not have DB or tag in it, exception is fine.
    std::string tagToken = metadata->getToken (tag);
    coral->commit();
    coral->disconnect ();

    // load the data from the db
    pool->connect ();
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
    pool->disconnect ();

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

}
