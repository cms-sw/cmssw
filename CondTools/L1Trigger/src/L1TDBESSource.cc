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
    : keyTag (iConfig.getParameter<std::string> ("tag")),
    reader (iConfig.getParameter<std::string> ("connect"), iConfig.getParameter<std::string> ("catalog"))
{
    // Register our data
    setWhatProduced(this, &L1TDBESSource::produceKey);
    setWhatProduced(this, &L1TDBESSource::produceCSC);
    findingRecord<L1TriggerKeyRcd> ();
    findingRecord<L1CSCTPParametersRcd> ();

    // load parameters to load
    // Check if we have field toGet and analyze it
    typedef std::vector<edm::ParameterSet> ToSave;
    ToSave toSave = iConfig.getParameter<ToSave> ("toLoad");
    for (ToSave::const_iterator it = toSave.begin (); it != toSave.end (); it++)
    {
        std::string record = it->getParameter<std::string> ("record");
        std::vector<std::string> recordItems = it->getParameter<std::vector<std::string> > ("data");

        // Copy items to the list items list
        Items::iterator rec = items.insert (std::make_pair (record, std::set<std::string> ())).first;
        for (std::vector<std::string>::const_iterator it = recordItems.begin (); it != recordItems.end (); it ++)
            rec->second.insert (*it);
    }
}

L1TDBESSource::~L1TDBESSource()
{
}

// ------------ method called to produce the data  ------------
std::auto_ptr<L1CSCTPParameters> L1TDBESSource::produceCSC(const L1CSCTPParametersRcd & iRecord)
{
    // first load the key for given value, then load the payload.
    // TODO: Should be implemented some kind of cache.
    L1TriggerKey key = reader.readKey (keyTag, iRecord.validityInterval ().first ().eventID ().run ());
    std::cerr << "L1TDBESSource: I found this key: " << key.getKey () << std::endl;
    
    L1CSCTPParameters csc = reader.readPayload<L1CSCTPParameters> (key);
    return std::auto_ptr<L1CSCTPParameters> (new L1CSCTPParameters (csc));
}

std::auto_ptr<L1TriggerKey> L1TDBESSource::produceKey(const L1TriggerKeyRcd& iRecord)
{
    // Load key from DB
    L1TriggerKey key = reader.readKey (keyTag, iRecord.validityInterval ().first ().eventID ().run ());
    return std::auto_ptr<L1TriggerKey> (new L1TriggerKey (key));
}

void L1TDBESSource::setIntervalFor (const edm::eventsetup::EventSetupRecordKey& a,
                                const edm::IOVSyncValue & b,
                                edm::ValidityInterval& oInterval)
{
    std::cerr << "L1TDBESSource: Looking for IOV at time: " << b.eventID ().run () << " for: "
        << a.type (). name () << std::endl;
    // I do not care which record - all my record will have the same IOV - the one that is
    // assigned to key
    std::pair<edm::RunNumber_t, edm::RunNumber_t> found = reader.interval (keyTag, b.eventID ().run ());
    if (found != DataReader::invalid ())
        oInterval = edm::ValidityInterval (edm:: IOVSyncValue (edm::EventID (found.first, 1)),
                edm::IOVSyncValue (edm::EventID (found.second, 1)));
    else
    {
        std::cerr << "L1TDBESSource: interval not found" << std::endl;
        oInterval = edm::ValidityInterval (edm::IOVSyncValue::invalidIOVSyncValue (),
                edm::IOVSyncValue::invalidIOVSyncValue ());
    }
}

}   // ns
