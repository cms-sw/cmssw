#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Ref.h"

#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVIterator.h"

#include "CondTools/L1Trigger/src/L1TDBESSource.h"

#include <iostream>

namespace l1t
{

//
// constructors and destructor
//
L1TDBESSource::L1TDBESSource(const edm::ParameterSet& iConfig)
{
    // Register our data
    setWhatProduced(this, &L1TDBESSource::produceKey);
    setWhatProduced(this, &L1TDBESSource::produceCSC);
    findingRecord<L1TriggerKeyRcd> ();
    findingRecord<L1CSCTPParametersRcd> ();

    // and connect to DB
    session = new cond::DBSession ();
    session->sessionConfiguration ().setAuthenticationMethod (cond::Env);
    session->connectionConfiguration ().enableConnectionSharing ();
    session->connectionConfiguration ().enableReadOnlySessionOnUpdateConnections ();
    session->open ();

    coral = new cond::RelationalStorageManager (iConfig.getParameter<std::string> ("connect"), session);
    metadata = new cond::MetaData (*coral);

    // Load tag token that is assigned for given tag
    coral->connect (cond::ReadOnly);
    coral->startTransaction (true);
    tagToken = metadata->getToken (iConfig.getParameter<std::string> ("tag"));
    std::cerr << "Tag token: \"" << tagToken  << "\"" << std::endl;
    coral->commit();
    coral->disconnect ();

    pool = new cond::PoolStorageManager (iConfig.getParameter<std::string> ("connect"),
            iConfig.getParameter<std::string> ("catalog") , session);

    // load all iovs into database
    loadIovMap ();
}

L1TDBESSource::~L1TDBESSource()
{
    // clean up the things
    delete metadata;
    delete coral;
    delete session;
    delete pool;
}

// ------------ method called to produce the data  ------------
std::auto_ptr<L1CSCTPParameters> L1TDBESSource::produceCSC(const L1CSCTPParametersRcd & iRecord)
{
    pool->connect ();
    pool->startTransaction (true);

    // Copy&paste -> bad. But I whant results fast:-)
    std::string keyValue = "Not set";

    // we can use any date from the interval provided by iRecord
    IntervalData data = intervals.find (iRecord.validityInterval ().first ().eventID ().run ());
    if (data != IntervalData::invalid ())
    {
        assert (data.start () == iRecord.validityInterval ().first ().eventID ().run ());
        assert (data.end () == iRecord.validityInterval ().last ().eventID ().run ());

        cond::Ref<L1TriggerKey> key (*pool, data.payload ());
        keyValue = std::string (key->getKey ());
    }

    std::cerr << "L1TKSoucre: I found this key: " << keyValue << std::endl;

    // so we have a key, let's use it to load data
    // Yeh, I need to load data from DB, and I do not have a payload token...
    cond::IOVService iov (*pool);
    cond::IOVIterator * iovIt = iov.newIOVIterator (metadata->getToken (keyValue));
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
    cond::Ref<L1CSCTPParameters> csc (*pool, data.payload ());

    pool->commit ();
    pool->disconnect ();
    return std::auto_ptr<L1CSCTPParameters> (new L1CSCTPParameters (*csc));
}

std::auto_ptr<L1TriggerKey> L1TDBESSource::produceKey(const L1TriggerKeyRcd& iRecord)
{
    pool->connect ();
    pool->startTransaction (true);

    std::string keyValue = "Not set";

    // we can use any date from the interval provided by iRecord
    IntervalData data = intervals.find (iRecord.validityInterval ().first ().eventID ().run ());
    if (data != IntervalData::invalid ())
    {
        assert (data.start () == iRecord.validityInterval ().first ().eventID ().run ());
        assert (data.end () == iRecord.validityInterval ().last ().eventID ().run ());

        cond::Ref<L1TriggerKey> key (*pool, data.payload ());
        keyValue = std::string (key->getKey ());
    }

    pool->commit ();
    pool->disconnect ();

    // first of all I need to load key from the "current" IOV
    // based on this key, I have to extract and return all other objects that are associated
    // this key and have valid (most likely infinitive IOV)
    return std::auto_ptr<L1TriggerKey> (new L1TriggerKey (keyValue));
}

void L1TDBESSource::setIntervalFor (const edm::eventsetup::EventSetupRecordKey& a,
                                const edm::IOVSyncValue & b,
                                edm::ValidityInterval& oInterval)
{
    std::cerr << "L1TKSoucre: Looking for IOV at time: " << b.eventID ().run () << " for: "
        << a.type (). name () << std::endl;
    // I do not care which record - all my record will have the same IOV - the one that is
    // assigned to key
    IntervalData found = intervals.find (b.eventID ().run ());
    if (found != IntervalData::invalid ())
        oInterval = edm::ValidityInterval (edm:: IOVSyncValue (edm::EventID (found.start (), 1)),
                edm::IOVSyncValue (edm::EventID (found.end (), 1)));
    else
    {
        std::cerr << "L1TKSoucre: interval not found" << std::endl;
        oInterval = edm::ValidityInterval (edm::IOVSyncValue::invalidIOVSyncValue (),
                edm::IOVSyncValue::invalidIOVSyncValue ());
    }
}

void L1TDBESSource::loadIovMap ()
{
    pool->connect ();
    pool->startTransaction (true);

    intervals.clear ();

    cond::IOVService iov (*pool);
    cond::IOVIterator * iovIt = iov.newIOVIterator (tagToken);

    while (iovIt->next ())
    {
        std::cerr << "L1TKSoucre: adding interval: " << iovIt->validity ().first
            << " -> " << iovIt->validity ().second << std::endl;
        intervals.addInterval (IntervalData (iovIt->validity ().first, iovIt->validity ().second, iovIt->payloadToken ()));
    }

    delete iovIt;

    pool->commit ();
    pool->disconnect ();
}

}   // ns
