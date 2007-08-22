// -*- C++ -*-
//
// Package:    CondTools/L1Trigger
// Class:      L1TWriter
// 
/**\class L1TWriter.cc CondTools/L1Trigger/src/L1TWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giedrius Bacevicius
//         Created:  Wed Jul 11 13:52:35 CEST 2007
// $Id: L1TWriter.cc,v 1.1 2007/08/20 16:25:17 giedrius Exp $
//
//


// system include files
#include <memory>
#include <sstream>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

#include "CondTools/L1Trigger/plugins/L1TWriter.h"

namespace l1t
{
    
//
// constructors and destructor
//
L1TWriter::L1TWriter(const edm::ParameterSet& iConfig)
    : sinceRun (-1), executionNumber (0),
    writer (iConfig.getParameter<std::string> ("connect"), iConfig.getParameter<std::string> ("catalog"))
{
    std::vector<std::string> params = iConfig.getParameterNames ();

    // key tag is required. There is no point to write data without key.
    keyTag = std::string (iConfig.getParameter<std::string> ("keyTag"));

    if (std::find (params.begin (), params.end (), std::string ("sinceRun")) != params.end ())
        sinceRun = iConfig.getParameter<int> ("sinceRun");

    // Load records that we have to save
    if (std::find (params.begin (), params.end (), std::string ("toSave")) != params.end ())
    {
        typedef std::vector<edm::ParameterSet> ToSave;
        ToSave toSave = iConfig.getParameter<ToSave> ("toSave");
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
}


L1TWriter::~L1TWriter()
{
}

// ------------ method called to for each event  ------------
void L1TWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // sanity check, to make sure we are not abusing system
    if (executionNumber != 0 && sinceRun > 0)
        // we can't have sinceRun set when running this for second time
        throw cond::Exception ("L1TWriter") << "We are executing this analyzer for the second time. "
            << "However, sinceRun parameter is set. This means that I have to save twice the same "
                << "L1 key for the same run.";

    executionNumber ++;
    // if we are using sinceRun, do not try to run this method more then once
    unsigned long long run = iEvent.id ().run ();
    if (sinceRun > 0)
        run = sinceRun;

    L1TriggerKey newKey;

    for (Items::const_iterator rIt = items.begin (); rIt != items.end (); rIt ++)
    {
        std::string record = rIt->first;
        std::set<std::string> types = rIt->second;

        for (std::set<std::string>::const_iterator tIt = types.begin (); tIt != types.end (); tIt++)
            writer.writePayload (newKey, iSetup, record, *tIt);
    }

    // save key to DB
    // here we need to insert current IOV into a list of previous ones
    // This can be easaly done via PoolDBService, but due to limited interface i have to do it myself...
    if (!keyTag.empty ())
        writer.writeKey (new L1TriggerKey (newKey), keyTag, run);

}

template<typename Record, typename Value>
void L1TWriter::writePayload (L1TriggerKey & key, const std::string & recordName, const edm::EventSetup & iSetup)
{
    edm::ESHandle<Value> handle;
    iSetup.get<Record> ().get (handle);

    writer.writePayload (key, new Value (*(handle.product ())), recordName);
}

}   // ns

