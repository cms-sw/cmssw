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

#include "CondTools/L1Trigger/plugins/L1TDBESSource.h"

#include "FWCore/Framework/interface/HCTypeTag.icc"
#include "FWCore/Framework/interface/HCTypeTagTemplate.h"

#include <iostream>

namespace l1t
{

//
// constructors and destructor
//
L1TDBESSource::L1TDBESSource(const edm::ParameterSet& iConfig)
    : keyTag (iConfig.getParameter<std::string> ("tag")),
    reader (iConfig.getParameter<std::string> ("connect"), iConfig.getParameter<std::string> ("catalog"))
{
    // HACK: Technically I should not require this line. But without it, I can't get L1TriggerKey to register.
    static edm::eventsetup::heterocontainer::HCTypeTagTemplate<L1TriggerKey, edm::eventsetup::DataKey> test;

    // make sure that we can provide information about the key as well
    registerRecord ("L1TriggerKeyRcd");

    // load parameters to load
    // Check if we have field toLoad and analyze it
    // Also make sure to register them as producer
    typedef std::vector<edm::ParameterSet> ToSave;
    ToSave toSave = iConfig.getParameter<ToSave> ("toLoad");

    for (ToSave::const_iterator it = toSave.begin (); it != toSave.end (); it++)
    {
        std::string record = it->getParameter<std::string> ("record");

        // Trigger Key will be loaded automaticlly, so ignore it here
        if (record == std::string ("L1TriggerKeyRcd"))
            continue;

        registerRecord (record);

        // Copy items to the list items list
        std::vector<std::string> recordItems = it->getParameter<std::vector<std::string> > ("data");
        Items::iterator rec = items.insert (std::make_pair (record, std::set<std::string> ())).first;
        for (std::vector<std::string>::const_iterator it = recordItems.begin (); it != recordItems.end (); it ++)
            rec->second.insert (*it);
    }
}

void L1TDBESSource::registerRecord (const std::string & record)
{
    static edm::eventsetup::EventSetupRecordKey defaultKey;
    edm::eventsetup::EventSetupRecordKey recordKey = edm::eventsetup::EventSetupRecordKey::TypeTag::findType(record);

    if (recordKey == defaultKey)
        throw cond::Exception ("L1TDBESSource::registerRecord") << "Type \"" << record << "\" was not found";

    // register record for production
    findingRecordWithKey (recordKey);
    usingRecordWithKey (recordKey);
}

L1TDBESSource::~L1TDBESSource()
{
}

void L1TDBESSource::setIntervalFor (const edm::eventsetup::EventSetupRecordKey& a,
                                const edm::IOVSyncValue & b,
                                edm::ValidityInterval& oInterval)
{
    // I do not care which record - all my record will have the same IOV - the one that is
    // assigned to key
    std::pair<edm::RunNumber_t, edm::RunNumber_t> found = reader.interval (keyTag, b.eventID ().run ());
    if (found != DataReader::invalid ())
        oInterval = edm::ValidityInterval (edm:: IOVSyncValue (edm::EventID (found.first, 1)),
                edm::IOVSyncValue (edm::EventID (found.second, 1)));
    else
        oInterval = edm::ValidityInterval (edm::IOVSyncValue::invalidIOVSyncValue (),
                edm::IOVSyncValue::invalidIOVSyncValue ());

    // check if this is a valid time, if so, update token, to load correct item
    if (b.eventID ().run () <= 0)
        return;

    if (a.name () == std::string ("L1TriggerKeyRcd"))
        reader.updateToken (a.name (), "L1TriggerKey", keyTag, b.eventID ().run ());
    else
    {
        // update token for all types associated with given record.
        // TODO: This can be optimized a bit, as all records will use the same key
        L1TriggerKey key = reader.readKey (keyTag, b.eventID ().run ());
        Items::const_iterator found = items.find (a.name ());

        std::set<std::string> toLoad = found->second;
        for (std::set<std::string>::const_iterator it = toLoad.begin (); it != toLoad.end (); it++)
            reader.updateToken (a.name (), *it, key.getKey (), b.eventID ().run ());
    }
}

void L1TDBESSource::newInterval(const edm::eventsetup::EventSetupRecordKey& recordKey, const edm::ValidityInterval& interval)
{
    invalidateProxies(recordKey);
}

void L1TDBESSource::registerProxies(const edm::eventsetup::EventSetupRecordKey& recordKey, KeyedProxies& proxyList)
{
    if (recordKey.name () == std::string ("L1TriggerKeyRcd"))
        registerKey (recordKey, proxyList);
    else
        registerPayload (recordKey, proxyList);
}

void L1TDBESSource::registerKey (const edm::eventsetup::EventSetupRecordKey& recordKey, KeyedProxies& proxyList)
{
    std::pair<boost::shared_ptr<edm::eventsetup::DataProxy>, edm::eventsetup::DataKey> proxy = reader.key ();
    proxyList.push_back (KeyedProxies::value_type (proxy.second, proxy.first));
}

void L1TDBESSource::registerPayload (const edm::eventsetup::EventSetupRecordKey& recordKey, KeyedProxies& proxyList)
{
    Items::const_iterator found = items.find (recordKey.name ());
    if (found == items.end ())
        throw cond::Exception ("L1TDBESSource::registerPayload") << "I was asked to produce record with name "
           << recordKey.name () << " but I have no idea what is that.";

    // register proxies for all types associated with given record.
    std::set<std::string> toLoad = found->second;
    for (std::set<std::string>::const_iterator it = toLoad.begin (); it != toLoad.end (); it++)
    {
        std::pair<boost::shared_ptr<edm::eventsetup::DataProxy>, edm::eventsetup::DataKey> proxy =
            reader.payload (recordKey.name (), *it);

        proxyList.push_back (KeyedProxies::value_type (proxy.second, proxy.first));
    }
}

}   // ns
